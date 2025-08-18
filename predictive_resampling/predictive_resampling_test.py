import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_resampling.predictive_resampling import (
    predictive_resampling_beta,
    predictive_resampling_beta_chunked
)
from models.model_config import ModelConfig
from models.model import unbin_y_values, bin_y_values


@pytest.fixture
def mock_config():
    """Create a mock ModelConfig for testing."""
    return ModelConfig(
        d_model=32,
        d_x=4,
        d_y=1,
        d_vocab=64,
        n_ctx=20,
        n_layers=2,
        y_min=-2.0,
        y_max=2.0
    )


@pytest.fixture
def mock_model(mock_config):
    """Create a mock model that returns predictable logits."""
    model = MagicMock()
    device = torch.device('cpu')
    
    # Create a proper parameter tensor with device
    param = torch.tensor([1.0], device=device)
    # Use a lambda to create a fresh iterator each time parameters() is called
    model.parameters.return_value = iter([param])
    model.parameters.side_effect = lambda: iter([param])
    
    # Mock forward pass to return logits with a specific pattern
    def mock_forward(x, y):
        batch_size, seq_len = x.shape[0], x.shape[1]
        # The real model creates sequences of length 2*seq_len due to construct_sequence
        output_seq_len = 2 * seq_len
        # Return logits that favor the middle bins
        logits = torch.zeros(batch_size, output_seq_len, mock_config.d_vocab, device=x.device)
        # Make middle bins more likely
        middle_bin = mock_config.d_vocab // 2
        logits[:, :, middle_bin-2:middle_bin+3] = 2.0
        return logits
    
    # Replace the mock's __call__ method to make model(x, y) work properly
    model.side_effect = mock_forward
    return model


class TestPredictiveResamplingBeta:
    
    def test_basic_functionality_no_init(self, mock_model, mock_config):
        """Test basic functionality without initialization tokens."""
        result = predictive_resampling_beta(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=3,
            sample_y=False
        )
        
        # Check output shape
        assert result.shape == (3, mock_config.d_x)
        assert isinstance(result, np.ndarray)
    
    def test_with_init_tokens(self, mock_model, mock_config):
        """Test functionality with initialization tokens."""
        batch_size = 1
        init_length = 2
        
        init_x = torch.randn(batch_size, init_length, mock_config.d_x)
        init_y = torch.randn(batch_size, init_length, mock_config.d_y)
        
        result = predictive_resampling_beta(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=2,
            sample_y=False,
            init_x=init_x,
            init_y=init_y
        )
        
        assert result.shape == (2, mock_config.d_x) #2 samples, d_x features for beta
        assert isinstance(result, np.ndarray)
    
    def test_sample_y_true(self, mock_model, mock_config):
        """Test with sampling enabled."""
        with patch('torch.multinomial') as mock_multinomial:
            # Mock multinomial to return predictable values
            mock_multinomial.return_value = torch.tensor([[32], [31]])  # Middle bins
            
            result = predictive_resampling_beta(
                model=mock_model,
                config=mock_config,
                forward_recursion_steps=3,
                forward_recursion_samples=2,
                sample_y=True
            )
            
            assert result.shape == (2, mock_config.d_x)
            assert mock_multinomial.called
    
    def test_sample_y_false(self, mock_model, mock_config):
        """Test with deterministic argmax."""
        result = predictive_resampling_beta(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=3,
            forward_recursion_samples=2,
            sample_y=False
        )
        
        assert result.shape == (2, mock_config.d_x)
    
    def test_init_tokens_validation(self, mock_model, mock_config):
        """Test validation of initialization tokens."""
        # Test with wrong batch size for init_x
        init_x = torch.randn(2, 3, mock_config.d_x)  # Batch size 2, should be 1
        init_y = torch.randn(2, 3, mock_config.d_y)
        
        with pytest.raises(AssertionError, match="init_x must have batch size 1"):
            predictive_resampling_beta(
                model=mock_model,
                config=mock_config,
                forward_recursion_steps=5,
                forward_recursion_samples=2,
                init_x=init_x,
                init_y=init_y
            )
    
    def test_init_tokens_length_mismatch(self, mock_model, mock_config):
        """Test validation of initialization token lengths."""
        init_x = torch.randn(1, 3, mock_config.d_x)
        init_y = torch.randn(1, 4, mock_config.d_y)  # Different length
        
        with pytest.raises(AssertionError, match="init_x and init_y must have the same length"):
            predictive_resampling_beta(
                model=mock_model,
                config=mock_config,
                forward_recursion_steps=6,
                forward_recursion_samples=2,
                init_x=init_x,
                init_y=init_y
            )
    
    def test_device_handling(self, mock_config):
        """Test that the function handles different devices correctly."""
        model = MagicMock()
        
        # Mock device as CUDA
        cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        param = torch.tensor([1.0], device=cuda_device)
        model.parameters.return_value = iter([param])
        
        def mock_forward(x, y):
            batch_size, seq_len = x.shape[0], x.shape[1]
            # The real model creates sequences of length 2*seq_len due to construct_sequence
            output_seq_len = 2 * seq_len
            logits = torch.zeros(batch_size, output_seq_len, mock_config.d_vocab, device=x.device)
            middle_bin = mock_config.d_vocab // 2
            logits[:, :, middle_bin] = 2.0
            return logits
        
        model.side_effect = mock_forward
        
        result = predictive_resampling_beta(
            model=model,
            config=mock_config,
            forward_recursion_steps=3,
            forward_recursion_samples=2,
            sample_y=False
        )
        
        assert result.shape == (2, mock_config.d_x)
        assert isinstance(result, np.ndarray)
    
    def test_empty_recursion_steps(self, mock_model, mock_config):
        """Test edge case with minimal recursion steps."""
        # This should handle the case where T-1 = K_init, so no iteration occurs
        init_x = torch.randn(1, 1, mock_config.d_x)
        init_y = torch.randn(1, 1, mock_config.d_y)
        
        result = predictive_resampling_beta(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=2,  # T=2, so only one prediction needed
            forward_recursion_samples=2,
            init_x=init_x,
            init_y=init_y
        )
        
        assert result.shape == (2, mock_config.d_x)


class TestPredictiveResamplingBetaChunked:
    
    def test_basic_chunking(self, mock_model, mock_config):
        """Test basic chunking functionality."""
        result = predictive_resampling_beta_chunked(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=10,
            chunk_size=3,
            sample_y=False
        )
        
        assert result.shape == (10, mock_config.d_x)
        assert isinstance(result, np.ndarray)
    
    def test_chunking_with_init_tokens(self, mock_model, mock_config):
        """Test chunking with initialization tokens."""
        init_x = torch.randn(1, 2, mock_config.d_x)
        init_y = torch.randn(1, 2, mock_config.d_y)
        
        result = predictive_resampling_beta_chunked(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=7,
            chunk_size=3,
            init_x=init_x,
            init_y=init_y
        )
        
        assert result.shape == (7, mock_config.d_x)
    
    def test_chunk_size_larger_than_samples(self, mock_model, mock_config):
        """Test when chunk size is larger than total samples."""
        result = predictive_resampling_beta_chunked(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=5,
            chunk_size=10,  # Larger than samples
            sample_y=False
        )
        
        assert result.shape == (5, mock_config.d_x)
    
    def test_exact_chunk_division(self, mock_model, mock_config):
        """Test when samples divide evenly into chunks."""
        result = predictive_resampling_beta_chunked(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=6,
            chunk_size=3,  # Divides evenly
            sample_y=False
        )
        
        assert result.shape == (6, mock_config.d_x)
    
    def test_init_tokens_validation_chunked(self, mock_model, mock_config):
        """Test validation of initialization tokens in chunked version."""
        init_x = torch.randn(2, 3, mock_config.d_x)  # Wrong batch size
        init_y = torch.randn(2, 3, mock_config.d_y)
        
        with pytest.raises(ValueError, match="init_x batch size \\(2\\) must be 1"):
            predictive_resampling_beta_chunked(
                model=mock_model,
                config=mock_config,
                forward_recursion_steps=5,
                forward_recursion_samples=4,
                chunk_size=2,
                init_x=init_x,
                init_y=init_y
            )
    
    @patch('predictive_resampling.predictive_resampling.predictive_resampling_beta')
    def test_chunked_calls_underlying_function(self, mock_func, mock_model, mock_config):
        """Test that chunked version properly calls the underlying function."""
        # Mock the underlying function to return predictable results
        mock_func.side_effect = [
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),  # First chunk
            np.array([[9, 10, 11, 12]])  # Second chunk
        ]
        
        result = predictive_resampling_beta_chunked(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=5,
            forward_recursion_samples=3,
            chunk_size=2
        )
        
        # Should call the function twice (chunks of 2 and 1)
        assert mock_func.call_count == 2
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))


class TestIntegrationWithRealModel:
    """Integration tests that use actual model components."""
    
    def test_unbin_y_values_integration(self, mock_config):
        """Test integration with actual unbin_y_values function."""
        # Test that bin indices are correctly converted back to y values
        bin_indices = torch.tensor([0, 31, 63])  # Min, middle, max bins
        y_values = unbin_y_values(bin_indices, mock_config.y_min, mock_config.y_max, mock_config.d_vocab)
        
        expected_min = mock_config.y_min + 0.5 * (mock_config.y_max - mock_config.y_min) / mock_config.d_vocab
        expected_max = mock_config.y_max - 0.5 * (mock_config.y_max - mock_config.y_min) / mock_config.d_vocab
        
        assert y_values.shape == (3, 1)
        assert torch.allclose(y_values[0], torch.tensor([[expected_min]]), rtol=1e-4)
        assert torch.allclose(y_values[2], torch.tensor([[expected_max]]), rtol=1e-4)
    
    def test_bin_unbin_roundtrip(self, mock_config):
        """Test that binning and unbinning is approximately consistent."""
        original_y = torch.tensor([[-1.5], [0.0], [1.5]])
        
        # Bin the values
        bin_indices = bin_y_values(original_y, mock_config.y_min, mock_config.y_max, mock_config.d_vocab)
        
        # Unbin them back
        recovered_y = unbin_y_values(bin_indices, mock_config.y_min, mock_config.y_max, mock_config.d_vocab)
        
        # Should be close (within half a bin width)
        bin_width = (mock_config.y_max - mock_config.y_min) / mock_config.d_vocab
        assert torch.allclose(original_y, recovered_y, atol=bin_width/2)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_samples(self, mock_model, mock_config):
        """Test with zero samples - should handle gracefully or raise appropriate error."""
        try:
            result = predictive_resampling_beta_chunked(
                model=mock_model,
                config=mock_config,
                forward_recursion_steps=5,
                forward_recursion_samples=0,
                chunk_size=2
            )
            assert result.shape == (0, mock_config.d_x)
        except (ValueError, RuntimeError):
            # This is acceptable behavior for zero samples
            pass
    
    def test_single_recursion_step(self, mock_model, mock_config):
        """Test with minimal recursion steps."""
        result = predictive_resampling_beta(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=1,
            forward_recursion_samples=2,
            sample_y=False
        )
        
        # Should still work, though the sequence will be very short
        assert result.shape == (2, mock_config.d_x)
    
    def test_large_init_sequence(self, mock_model, mock_config):
        """Test with initialization sequence that's close to total length."""
        init_length = 8
        total_steps = 10
        
        init_x = torch.randn(1, init_length, mock_config.d_x)
        init_y = torch.randn(1, init_length, mock_config.d_y)
        
        result = predictive_resampling_beta(
            model=mock_model,
            config=mock_config,
            forward_recursion_steps=total_steps,
            forward_recursion_samples=2,
            init_x=init_x,
            init_y=init_y
        )
        
        assert result.shape == (2, mock_config.d_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])