import os
import json
import torch
import numpy as np
from tempfile import TemporaryDirectory
from nam.train.core import train, get_wavenet_config, Architecture
from nam.models.wavenet import WaveNet

# Helper to create a dummy wav file (float32, mono)
def create_dummy_wav(path, length=16000, value=0.1):
    import soundfile as sf
    data = np.full((length,), value, dtype=np.float32)
    sf.write(path, data, 48000)

def main():
    with TemporaryDirectory() as tmpdir:
        # Set ny and calculate nx based on receptive field
        ny = 8192
        # Use the same architecture as in train()
        arch = Architecture.STANDARD
        wavenet_config = get_wavenet_config(arch)
        # Patch condition_size for dummy dataset (2 knob types + 1 level)
        condition_size = 3
        for layer in wavenet_config['layers_configs']:
            layer['condition_size'] = condition_size
        model = WaveNet(**wavenet_config)
        rf = model.receptive_field
        nx = ny + rf - 1
        print(f"Using nx={nx}, ny={ny}, receptive_field={rf}")
        # Calculate required dummy wav length
        dummy_length = nx + ny - 1 + 1000
        # Create dummy audio files
        input1 = os.path.join(tmpdir, 'input1.wav')
        output1 = os.path.join(tmpdir, 'output1.wav')
        input2 = os.path.join(tmpdir, 'input2.wav')
        output2 = os.path.join(tmpdir, 'output2.wav')
        create_dummy_wav(input1, length=dummy_length, value=0.1)
        create_dummy_wav(output1, length=dummy_length, value=0.2)
        create_dummy_wav(input2, length=dummy_length, value=0.3)
        create_dummy_wav(output2, length=dummy_length, value=0.4)
        # Create JSON dataset
        json_path = os.path.join(tmpdir, 'dataset.json')
        dataset = [
            {"input_path": input1, "output_path": output1, "knob_type": "gain", "knob_level": 0.7},
            {"input_path": input2, "output_path": output2, "knob_type": "bass", "knob_level": 0.3},
        ]
        with open(json_path, 'w') as f:
            json.dump(dataset, f)
        # Run training for 1 epoch, small batch
        result = train(
            dataset_type='json_conditioned',
            json_path=json_path,
            train_path=tmpdir,
            epochs=1,
            nx=nx,
            ny=ny,
            batch_size=1,
            lr=0.001,
            lr_decay=0.01,
        )
        print('Training complete. Model:', result.model)

if __name__ == "__main__":
    main() 