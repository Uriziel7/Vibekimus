"""Units of time and size, defined once for the whole program. Standard library only
(nothing at all imported): the .exe's launcher may use it before any library is installed."""

MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE

# Binary sizes, as Windows shows them (file sizes, graphics memory).
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * BYTES_PER_KB
BYTES_PER_GB = 1024 * BYTES_PER_MB
# Decimal sizes, as downloads and disks are labelled.
BYTES_PER_DECIMAL_MB = 1_000_000
BYTES_PER_DECIMAL_GB = 1_000_000_000
