
# Imports
import wave
import numpy as np


# Write curve
def write_curve(
        file_path: str,
        raw_data: np.array,
        sample_rate: float = 44100.0,
):
    r"""

    Args:
        file_path:

    Returns:

    """
    # Open
    with wave.open(file_path, 'wb') as w_file:
        # Properties
        w_file.setframerate(sample_rate)
        w_file.setnchannels(1)
        w_file.setsampwidth(4)
        w_file.writeframes(raw_data.tobytes())
    # end with
# end write_curve
