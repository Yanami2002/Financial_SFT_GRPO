"""Train the FinQA DAPO adaptation from the selected SFT adapter."""
from train.grpo import main

if __name__ == '__main__':
    main(dapo=True)
