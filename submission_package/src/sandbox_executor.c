/*
 * sandbox_executor.c
 * Safe execution sandbox for running Python code and capturing outputs.
 * Uses fork + exec + timeout to isolate execution.
 * Returns JSON with exit status, stdout, stderr, and timing.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>

#define MAX_OUTPUT_SIZE 1024 * 1024  // 1MB max output
#define DEFAULT_TIMEOUT_SEC 5
#define READ_END 0
#define WRITE_END 1

/* Read all data from a file descriptor */
ssize_t read_all(int fd, char *buffer, size_t max_size) {
    size_t total = 0;
    ssize_t n;
    while (total < max_size - 1) {
        n = read(fd, buffer + total, max_size - total - 1);
        if (n <= 0) break;
        total += n;
    }
    buffer[total] = '\0';
    return total;
}

/* Escape JSON string */
void escape_json(const char *input, char *output, size_t max_out) {
    size_t i = 0, j = 0;
    while (input[i] != '\0' && j < max_out - 2) {
        switch (input[i]) {
            case '"':  output[j++] = '\\'; output[j++] = '"'; break;
            case '\\': output[j++] = '\\'; output[j++] = '\\'; break;
            case '\n': output[j++] = '\\'; output[j++] = 'n'; break;
            case '\r': output[j++] = '\\'; output[j++] = 'r'; break;
            case '\t': output[j++] = '\\'; output[j++] = 't'; break;
            default:
                if (input[i] < 32) {
                    j += sprintf(output + j, "\\u%04x", input[i]);
                } else {
                    output[j++] = input[i];
                }
        }
        i++;
    }
    output[j] = '\0';
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("{\"error\": \"Usage: %s <python_code>\\n\", \"exit_code\": -1}\n", argv[0]);
        return 1;
    }

    const char *code = argv[1];
    int timeout_sec = DEFAULT_TIMEOUT_SEC;
    if (argc >= 3) {
        timeout_sec = atoi(argv[2]);
    }

    int stdout_pipe[2];
    int stderr_pipe[2];
    int status;

    if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        printf("{\"error\": \"Failed to create pipes\\n\", \"exit_code\": -1}\n");
        return 1;
    }

    pid_t pid = fork();

    if (pid == -1) {
        printf("{\"error\": \"Fork failed\\n\", \"exit_code\": -1}\n");
        return 1;
    }

    if (pid == 0) {
        /* Child process */
        close(stdout_pipe[READ_END]);
        close(stderr_pipe[READ_END]);

        /* Redirect stdout and stderr to pipes */
        dup2(stdout_pipe[WRITE_END], STDOUT_FILENO);
        dup2(stderr_pipe[WRITE_END], STDERR_FILENO);
        close(stdout_pipe[WRITE_END]);
        close(stderr_pipe[WRITE_END]);

        /* Execute Python code */
        execlp("python3", "python3", "-c", code, (char *)NULL);

        /* If exec fails */
        fprintf(stderr, "Execution failed: %s\n", strerror(errno));
        _exit(127);
    }

    /* Parent process */
    close(stdout_pipe[WRITE_END]);
    close(stderr_pipe[WRITE_END]);

    /* Set up timeout */
    struct timeval start, current;
    gettimeofday(&start, NULL);

    int timed_out = 0;
    while (1) {
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            break;  /* Child exited */
        }
        if (result == -1) {
            break;  /* Error */
        }
        gettimeofday(&current, NULL);
        long elapsed_ms = (current.tv_sec - start.tv_sec) * 1000 +
                          (current.tv_usec - start.tv_usec) / 1000;
        if (elapsed_ms > timeout_sec * 1000) {
            timed_out = 1;
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }
        usleep(10000);  /* Sleep 10ms */
    }

    /* Read output */
    char stdout_buf[MAX_OUTPUT_SIZE];
    char stderr_buf[MAX_OUTPUT_SIZE];
    ssize_t stdout_len = read_all(stdout_pipe[READ_END], stdout_buf, MAX_OUTPUT_SIZE);
    ssize_t stderr_len = read_all(stderr_pipe[READ_END], stderr_buf, MAX_OUTPUT_SIZE);
    close(stdout_pipe[READ_END]);
    close(stderr_pipe[READ_END]);

    /* Determine exit code */
    int exit_code;
    if (timed_out) {
        exit_code = -1;  /* Timeout */
    } else if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);
    } else {
        exit_code = -2;
    }

    /* Build JSON output */
    char escaped_stdout[MAX_OUTPUT_SIZE * 2];
    char escaped_stderr[MAX_OUTPUT_SIZE * 2];
    escape_json(stdout_buf, escaped_stdout, MAX_OUTPUT_SIZE * 2);
    escape_json(stderr_buf, escaped_stderr, MAX_OUTPUT_SIZE * 2);

    printf("{\n");
    printf("  \"exit_code\": %d,\n", exit_code);
    printf("  \"timed_out\": %s,\n", timed_out ? "true" : "false");
    printf("  \"stdout\": \"%s\",\n", escaped_stdout);
    printf("  \"stderr\": \"%s\",\n", escaped_stderr);
    printf("  \"stdout_len\": %ld,\n", (long)stdout_len);
    printf("  \"stderr_len\": %ld\n", (long)stderr_len);
    printf("}\n");

    return 0;
}
