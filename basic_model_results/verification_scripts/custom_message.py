from aes_model_comparison import test_specific_plaintexts

test_results = test_specific_plaintexts(
    model_path="aes_nn_models/aes_cnn_model_e99_vl0.0070.keras",
    plaintexts=[
        b'This is a test!!',
        b'Another example.',
        b'Hello, world!!!',
    ],
    keys=b'SuperSecretKey!!'  # Same key for all plaintexts
)
