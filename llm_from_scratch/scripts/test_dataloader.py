from llm_from_scratch.pretraining.data.data_loader import create_dataloaders


def main() -> None:
    loaders = create_dataloaders(batch_size=4, max_length=256, stride=128)

    for split, loader in loaders.items():
        input_batch, target_batch = next(iter(loader))
        print(f"[{split}] windows: {len(loader.dataset)}, input shape: {input_batch.shape}")


if __name__ == "__main__":
    main()
