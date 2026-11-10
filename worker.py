import os
import sys
import time
from pathlib import Path

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = Path("checkpoint.pt")


def make_model():
    torch.manual_seed(71)
    return torch.nn.Linear(16, 4).to(DEVICE)


def run_short_job():
    model = make_model()
    with open("starts", "a", encoding="utf-8") as log:
        print(time.monotonic(), file=log)
    with torch.inference_mode():
        model(torch.ones(1, 16, device=DEVICE)).cpu()
    print(f"Short inference job finished on {DEVICE}", flush=True)


def run_training(mode):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    start = 0

    if mode == "resume":
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        print(f"Resuming after step {start}", flush=True)

    for step in range(start + 1, 101):
        generator = torch.Generator().manual_seed(step)
        batch = torch.randn(32, 16, generator=generator).to(DEVICE)
        target = batch[:, :4] * 2
        optimizer.zero_grad()
        (model(batch) - target).square().mean().backward()
        optimizer.step()

        if step == 1:
            Path("active").touch()
        time.sleep(0.05)

        if Path("pause").exists():
            torch.save(
                {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                "pending.pt",
            )
            os.replace("pending.pt", CHECKPOINT)
            print(f"Checkpoint committed after step {step}", flush=True)
            raise SystemExit(75)

    torch.save(model.state_dict(), f"{mode}.pt")
    Path("finished").write_text(str(time.monotonic()), encoding="utf-8")


if __name__ == "__main__":
    selected_mode = sys.argv[1]
    if selected_mode == "short":
        run_short_job()
    elif selected_mode in {"train", "resume", "reference"}:
        run_training(selected_mode)
    else:
        raise SystemExit(f"Unknown worker mode: {selected_mode}")
