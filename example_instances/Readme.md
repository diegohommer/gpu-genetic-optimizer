# Format

The files have the following format. The line numbers below refer to non-empty lines.

- **Line 1:** Number of GPUs (`n`).
- **Line 2:** Amount of VRAM (`V`) (the same for all GPUs).
- **Line 3:** Number of different types (`|T|`).
- **Line 4:** Number of PRNs (`m`).
- **Lines 5 to (m+4):** Each of the `m` PRNs is described on one line. Each line contains, in this order, the following values (as positive integers separated by spaces):
  - PRN type (`t_j`) (value from `1` to `|T|`).
  - VRAM consumption (`v_j`) (value from `1` to `V`, usually much smaller than `V`).
