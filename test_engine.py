from scripts.sit_app.inference import SiTGenerator
import os

# Define paths
ckpt = "SiT/weights/results-gene0error-fix/000-SiT-XL-2-Linear-velocity-None/checkpoints/0110000.pt"
mapping = "data/class_mapping.json"

# 1. Initialize (This should take a few seconds)
engine = SiTGenerator(ckpt, mapping)

# 2. Generate
print("Generating image...")
img = engine.generate("ABCA4", "L", 65)

# 3. Save to verify
img.save("test_output.png")
print("Success, Saved test_output.png")