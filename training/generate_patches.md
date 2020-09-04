# Create train datasets
Place L1C and L2A data into train folder, and then run:
```bash
python generate_patches.py L1C --save_prefix "../data/l1c" --run_60 --train_data
python generate_patches.py L1C --save_prefix "../data/l1c" --train_data
python generate_patches.py L2A --save_prefix "../data/l2a" --run_60 --train_data
python generate_patches.py L2A --save_prefix "../data/l2a" --train_data
```
Create validation set:
```bash
python training/create_validation_set.py --path "l1ctrain/"
python training/create_validation_set.py --path "l1ctrain60/" --run_60
python training/create_validation_set.py --path "l2atrain/"
python training/create_validation_set.py --path "l2atrain60/" --run_60
```
# Create test datasets
```bash
python generate_patches.py L1C_test --save_prefix "l1c" --run_60 --test_data
python generate_patches.py L1C_test --save_prefix "l1c" --test_data
python generate_patches.py L2A_test --save_prefix "../data/l2a" --run_60 --test_data
python generate_patches.py L2A_test --save_prefix "../data/l2a" --test_data
```

# Upload datasets
Test:
```bash
aws s3 cp --recursive l1ctest s3://s2-l1c-training-imgs/v2/l1ctest/
aws s3 cp --recursive l1ctest60 s3://s2-l1c-training-imgs/v2/l1ctest60/
aws s3 cp --recursive l2atest s3://s2-l1c-training-imgs/v2/l2atest/
aws s3 cp --recursive l2atest60 s3://s2-l1c-training-imgs/v2/l2atest60/
```
Train:
```bash
aws s3 cp --recursive l1ctrain s3://s2-l1c-training-imgs/v2/l1ctrain/
aws s3 cp --recursive l1ctrain60 s3://s2-l1c-training-imgs/v2/l1ctrain60/
aws s3 cp --recursive l2atrain s3://s2-l1c-training-imgs/v2/l2atrain/
aws s3 cp --recursive l2atrain60 s3://s2-l1c-training-imgs/v2/l2atrain60/
```
