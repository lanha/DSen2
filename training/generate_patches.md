# Create train datasets
Place L1C and L2A data into train folder, and then run:
```bash
python generate_patches.py L1C --save_prefix "../data/l1c" --run_60 --train_data
python generate_patches.py L1C --save_prefix "../data/l1c" --train_data
python generate_patches.py L2A --save_prefix "../data/l2a" --run_60 --train_data
python generate_patches.py L2A --save_prefix "../data/l2a" --train_data
```

# Create test datasets
```bash
python generate_patches.py L1C_test --save_prefix "../data/l1c" --run_60 --test_data
python generate_patches.py L1C_test --save_prefix "../data/l1c" --test_data
python generate_patches.py L2A_test --save_prefix "../data/l2a" --run_60 --test_data
python generate_patches.py L2A_test --save_prefix "../data/l2a" --test_data
```
