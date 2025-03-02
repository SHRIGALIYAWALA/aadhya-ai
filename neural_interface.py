import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

class BCIController:
    def __init__(self, board_id=0):
        """Initialize the Brain-Computer Interface (BCI) controller."""
        self.board_id = board_id
        self.params = BrainFlowInputParams()
        self.board = BoardShim(self.board_id, self.params)

    def connect(self):
        """Connect to the EEG headset and start streaming data."""
        self.board.prepare_session()
        self.board.start_stream()
        print("[BCI] Connection established. Streaming EEG data...")

    def process_brain_signals(self):
        """Process EEG signals and interpret user intent."""
        data = self.board.get_board_data()
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        if data.shape[1] > 0:
            for channel in eeg_channels:
                DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(self.board_id), 5.0, 50.0, 2, FilterTypes.BUTTERWORTH, 0)
                DataFilter.perform_wavelet_denoising(data[channel], "db4", 3)
            
            alpha_power = np.mean(data[eeg_channels[0]])  # Example: Using the first EEG channel
            if alpha_power > 0.5:
                print("[BCI] Detected Focus Mode: Activating AI...")
            elif alpha_power < -0.5:
                print("[BCI] Detected Relaxation Mode: Reducing AI Activity...")
            else:
                print("[BCI] No significant mental state detected.")

    def disconnect(self):
        """Stop streaming and disconnect the BCI device."""
        self.board.stop_stream()
        self.board.release_session()
        print("[BCI] Disconnected from EEG headset.")

# Example usage
if __name__ == "__main__":
    bci = BCIController()
    bci.connect()
    bci.process_brain_signals()
    bci.disconnect()
