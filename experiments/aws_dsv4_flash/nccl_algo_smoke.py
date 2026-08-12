import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))

    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {"rank": rank})
    assert [item["rank"] for item in gathered] == list(range(dist.get_world_size()))

    payload = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    if rank == 0:
        payload.fill_(37)
    dist.broadcast(payload, src=0)
    assert payload[0].item() == 37 and payload[-1].item() == 37

    dist.barrier()
    if rank == 0:
        print(f"NCCL smoke passed: algo={os.environ.get('NCCL_ALGO')} world_size={dist.get_world_size()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
