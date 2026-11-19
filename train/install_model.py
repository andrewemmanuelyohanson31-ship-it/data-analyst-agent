"""Register the fine-tuned GGUF from Colab with Ollama, using qwen2.5:3b's own chat template (tool calls included).

    python train/install_model.py C:\\Users\\me\\Downloads\\daa-qwen.Q4_K_M.gguf [--name daa-qwen]

Then: python eval/run_eval.py --model daa-qwen --no-parser   (and type daa-qwen in the app's model box)
"""
import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gguf")
    p.add_argument("--name", default="daa-qwen")
    p.add_argument("--base", default="qwen2.5:3b", help="the model whose template the training data copied")
    a = p.parse_args()
    gguf = Path(a.gguf).resolve()
    if not gguf.exists():
        raise SystemExit(f"Not found: {gguf}")
    template = subprocess.run(["ollama", "show", a.base, "--template"], capture_output=True, text=True,
                              encoding="utf-8", check=True).stdout.strip()
    modelfile = gguf.with_name("Modelfile")
    modelfile.write_text(f'FROM {gguf}\nTEMPLATE """{template}"""\nPARAMETER stop "<|im_end|>"\n', encoding="utf-8")
    subprocess.run(["ollama", "create", a.name, "-f", str(modelfile)], check=True)
    print(f"Installed '{a.name}'. Measure it: python eval/run_eval.py --model {a.name} --no-parser")


if __name__ == "__main__":
    main()
