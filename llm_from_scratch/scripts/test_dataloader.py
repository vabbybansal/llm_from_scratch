from llm_from_scratch.libs.data_loader import create_dataloader


def main() -> None:
    dataloader = create_dataloader(batch_size=4, max_length=256, stride=128)

    print(f"dataset windows: {len(dataloader.dataset)}")
    print(f"batch size: {dataloader.batch_size}")

    input_batch, target_batch = next(iter(dataloader))

    print(f"input batch shape: {input_batch.shape}")
    print(f"target batch shape: {target_batch.shape}")
    print(f"first input ids: {input_batch[0][:10].tolist()}")
    print(f"first target ids: {target_batch[0][:10].tolist()}")


if __name__ == "__main__":
    main()
