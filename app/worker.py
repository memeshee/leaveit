"""Leave It worker. Kill this process; the fuses do not care."""
import asyncio, sys

from temporalio.client import Client
from temporalio.worker import Worker

import workflows

TASK_QUEUE = "leaveit"


async def main() -> None:
    client = await Client.connect("localhost:7233")
    worker = Worker(client, task_queue=TASK_QUEUE,
                    workflows=[workflows.SealWorkflow],
                    activities=[workflows.record_seal, workflows.resurface, workflows.rescue])
    print("worker up", flush=True)
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
