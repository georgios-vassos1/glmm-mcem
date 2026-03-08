---
name: check
description: Run the glmm-mcem test suite and inject coverage + source layout for repo context
disable-model-invocation: true
---

## Source layout
```
!`find src tests -type f -name "*.py" | sort`
```

## Test results + coverage
```
!`uv run pytest --cov=glmm_mcem --cov-report=term-missing --tb=short -q 2>&1`
```

Review the above output. Summarise:
1. Which tests passed / failed (if any).
2. Any uncovered lines that look worth addressing.
3. Whether anything looks broken or suspicious.

Then ask what I'd like to work on next.
