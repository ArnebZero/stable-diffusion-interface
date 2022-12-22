import os
import re
import torch
import logging
import kfserving
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torch.cuda.amp import autocast
from contextlib import nullcontext
from logging.handlers import TimedRotatingFileHandler

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class CustomFormatter(logging.Formatter):
    format_pattern = "[{level} %(asctime)s %(pathname)s:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: format_pattern.format(level="D"),
        logging.INFO: format_pattern.format(level="I"),
        logging.ERROR: format_pattern.format(level="E"),
        logging.WARNING: format_pattern.format(level="W"),
        logging.CRITICAL: format_pattern.format(level="C"),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%y%m%d %H:%M:%S")

        if "pathname" in record.__dict__ and isinstance(record.pathname, str):
            # truncate the pathname
            filename = os.path.basename(record.pathname)
            filename = os.path.splitext(filename)[0]

            if len(filename) > 20:
                filename = f"{filename[:3]}~{filename[-16:]}"
            record.pathname = filename

        return formatter.format(record)


class KFStableDiffusionModel(kfserving.KFModel):
    def __init__(self, name: str, conf_path: str):
        super().__init__(name)
        self.ready = False

        self.opt = OmegaConf.load(conf_path)

        if self.opt.laion400m:
            self.opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            self.opt.ckpt = "models/ldm/text2img-large/model.ckpt"

        self.config = OmegaConf.load(f"{self.opt.config}")

        self.model = None
        self.logger = None

    def load(self):
        self.logger = logging.getLogger(__name__)
        log_handler = TimedRotatingFileHandler("logs/app.log", backupCount=24)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(log_handler)

        self.model = self.load_model_from_config(self.config, f"{self.opt.ckpt}")
        self.model = self.model.half()

        self.ready = True

    def predict(self, request):
        data = request["data"]

        output = []

        for item in data:
            try:
                user_id = item["id"]
                text = item["text"]
                assert isinstance(text, str)
                assert len(text) <= 100
            except:
                self.logger.error("Can't parse data")
                continue

            try:
                images = self.generate(text)
                output.append(
                    {
                        "id": user_id,
                        "images": {
                            str(ind): img.flatten().tolist()
                            for ind, img in enumerate(images)
                        },
                    }
                )
            except:
                self.logger.error(f"Can't create images for text '{text}'")
                output.append({"id": user_id, "error": "Can't create images"})

        return {"data": output, "result": len(output)}

    @staticmethod
    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            self.logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            self.logger.warning("missing keys:")
            self.logger.warning(m)
        if len(u) > 0 and verbose:
            self.logger.warning("unexpected keys:")
            self.logger.warning(u)

        model.cuda()
        model.eval()
        return model

    def generate(self, prompt):
        device = "cuda"
        if self.opt.plms:
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)
        images = []

        batch_size = self.opt.n_samples
        if not self.opt.from_file:
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            self.logger.info(f"reading prompts from {self.opt.from_file}")
            with open(self.opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(self.chunk(data, batch_size))

        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn(
                [
                    self.opt.n_samples,
                    self.opt.C,
                    self.opt.H // self.opt.f,
                    self.opt.W // self.opt.f,
                ],
                device=device,
            )

        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope():
                with self.model.ema_scope():
                    for _ in trange(self.opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.opt.scale != 1.0:
                                uc = self.model.get_learned_conditioning(
                                    batch_size * [""]
                                )
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [
                                self.opt.C,
                                self.opt.H // self.opt.f,
                                self.opt.W // self.opt.f,
                            ]
                            samples_ddim, _ = sampler.sample(
                                S=self.opt.ddim_steps,
                                conditioning=c,
                                batch_size=self.opt.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=self.opt.scale,
                                unconditional_conditioning=uc,
                                eta=self.opt.ddim_eta,
                                x_T=start_code,
                            )

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                            )

                            for x_sample in x_samples_ddim:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                images += [x_sample.astype(np.uint8)]

                    return images


if __name__ == "__main__":
    hostname = os.environ.get("HOSTNAME")
    x = re.compile(r"(kfserving-\d+)").search(hostname) if hostname else None

    service_name = x[0] if x else "kfserving-default"

    model = KFStableDiffusionModel(name=service_name, conf_path="config.yaml")
    model.load()
    kfserving.KFServer(workers=1).start([model])
