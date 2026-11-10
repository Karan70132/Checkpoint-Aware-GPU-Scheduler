#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

pid_t launch(const char* mode) {
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork failed");
    }
    if (pid == 0) {
        execlp("python3", "python3", "src/worker.py", mode, nullptr);
        _exit(127);
    }
    return pid;
}

int process_status(pid_t pid, bool block) {
    int value = 0;
    pid_t result;
    do {
        result = waitpid(pid, &value, block ? 0 : WNOHANG);
    } while (result < 0 && errno == EINTR);

    if (result < 0) {
        throw std::runtime_error("waitpid failed");
    }
    if (result == 0) {
        return -1;
    }
    return WIFEXITED(value) ? WEXITSTATUS(value) : 128;
}

void drain_queue() {
    while (!fs::is_empty("queue")) {
        fs::path next;
        for (const auto& entry : fs::directory_iterator("queue")) {
            if (next.empty() || entry.path() < next) {
                next = entry.path();
            }
        }
        fs::remove(next);
        if (process_status(launch("short"), true) != 0) {
            throw std::runtime_error("inference worker failed");
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "fcfs" && std::string(argv[1]) != "checkpoint")) {
        std::cerr << "Usage: ./scheduler {fcfs|checkpoint}\n";
        return 2;
    }

    fs::create_directories("queue");
    const bool checkpoint_policy = std::string(argv[1]) == "checkpoint";
    pid_t trainer = launch("train");
    int code = -1;

    while ((code = process_status(trainer, false)) == -1) {
        if (checkpoint_policy && !fs::is_empty("queue") && !fs::exists("pause")) {
            std::ofstream{"pause"};
        }
        std::this_thread::sleep_for(5ms);
    }

    if (code != 0 && code != 75) {
        return 1;
    }
    std::ofstream{"training_exit"} << code;
    drain_queue();

    if (code == 75) {
        fs::remove("pause");
        return process_status(launch("resume"), true) == 0 ? 0 : 1;
    }
    return 0;
}
