# Architecture notes

The scheduler launches workers with `fork` and `exec`, observes them with `waitpid`, and uses an exit code of 75 to distinguish a cooperative checkpoint from ordinary completion. The queue is represented by files ordered lexicographically. This keeps the prototype observable and easy to reproduce in Colab.

The training worker checks for a pause request between optimizer steps. It stores the completed step, model state, and optimizer state in a temporary file. `os.replace` atomically publishes the new checkpoint. The process then exits, releasing CUDA context ownership before the inference worker starts.

After draining the short-job queue, the scheduler starts a fresh worker in resume mode. The correctness notebook compares its final weights with an uninterrupted reference. The benchmark notebook compares first-come, first-served behavior against cooperative checkpoint scheduling.

## Production extensions

1. Replace queue files with a Unix-domain socket protocol and durable metadata store.
2. Record user identity, resource requirements, arrival time, priority, and maximum wait.
3. Use priority aging to prevent starvation.
4. Confirm device-memory release before dispatching the next worker.
5. Persist logs and scheduler state across scheduler crashes.
6. Support multi-GPU placement and memory-aware admission control.
7. Benchmark checkpoint cost, process startup, inference duration, and restart cost separately.
