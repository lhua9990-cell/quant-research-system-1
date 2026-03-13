from __future__ import annotations

import argparse


def additional_shares_to_limit_loss(
    cost_price: float,
    shares: float,
    current_price: float,
    loss_limit: float = 0.33,
) -> float:
    if cost_price <= 0 or shares <= 0 or current_price <= 0:
        raise ValueError("cost_price, shares, current_price must be positive")
    if loss_limit <= 0 or loss_limit >= 1:
        raise ValueError("loss_limit must be between 0 and 1")
    loss_rate = (current_price - cost_price) / cost_price
    if loss_rate >= -loss_limit:
        return 0.0
    target_avg = current_price / (1 - loss_limit)
    denom = current_price - target_avg
    if denom == 0:
        return 0.0
    additional = shares * (target_avg - cost_price) / denom
    return max(0.0, additional)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cost_price", type=float)
    parser.add_argument("shares", type=float)
    parser.add_argument("current_price", type=float)
    parser.add_argument("--loss-limit", type=float, default=0.33)
    args = parser.parse_args()
    additional = additional_shares_to_limit_loss(
        cost_price=args.cost_price,
        shares=args.shares,
        current_price=args.current_price,
        loss_limit=args.loss_limit,
    )
    print(additional)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
