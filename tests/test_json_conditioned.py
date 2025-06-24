import os
import json
import torch
import numpy as np
from tempfile import TemporaryDirectory
from nam.train.core import train

# Helper to create a dummy wav file (float32, mono)
def create_dummy_wav(path, length=16000, value=0.1):
    import soundfile as sf
    data = np.full((length,), value, dtype=np.float32)
    sf.write(path, data, 48000)


def test_json_conditioned_training():
    with TemporaryDirectory() as tmpdir:
        # Create dummy audio files
        input1 = os.path.join(tmpdir, 'input1.wav')
        output1 = os.path.join(tmpdir, 'output1.wav')
        input2 = os.path.join(tmpdir, 'input2.wav')
        output2 = os.path.join(tmpdir, 'output2.wav')
        create_dummy_wav(input1, value=0.1)
        create_dummy_wav(output1, value=0.2)
        create_dummy_wav(input2, value=0.3)
        create_dummy_wav(output2, value=0.4)
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
            nx=1024,
            ny=1,
            batch_size=1,
            lr=0.001,
            lr_decay=0.01,
        )
        assert result is not None
        assert hasattr(result, 'model')
        print('Test passed: Model trained with JSON-conditioned dataset.') 