"""Capture the CLI shots for the README, from real commands against a real backend.

The dashboard shot has `capture_dashboard.py`. This is the other half: the three moments the
README claims Pharos exists for — a prompt that does not fit, the plan that makes it fit, and
the run's own verdict on what it then did.

Nothing here is staged. The script builds a small but genuine project (a `shop/` package that
formats with `%`, a test suite, a clean `ruff` baseline), writes a `pharos.toml` beside it with
a deliberately small `num_ctx`, and then runs the ordinary CLI against it as a subprocess.
Whatever those commands print is what the SVG shows, including a bad verdict.

    uv run python docs/capture_cli.py                 # all three
    uv run python docs/capture_cli.py --only check,split
    uv run python docs/capture_cli.py --keep          # leave the fixture on disk

`--only check,split` needs a backend for the budget but never generates; `run` needs a model
that can call tools (`qwen3.5:9b` here — see DESKTOP_VALIDATION.md §16) and takes 10-20 minutes.

Colour is the one thing that has to be forced. Rich writes ANSI only to a terminal, and on
Windows a redirected stream is detected as a legacy console and gets no escapes at all, so a
piped capture comes out as flat grey text. The child process therefore starts with
`detect_legacy_windows` pinned to False and `FORCE_COLOR` set. That is a lie told to Rich about
the destination, and nothing else: the words, the numbers and the exit codes are the CLI's own.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import MONOKAI
from rich.text import Text

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
WIDTH = 100

# Small enough that the prompt below cannot fit, large enough that a part can do real work.
# The whole argument of the project is what happens between those two facts.
SHOT_NUM_CTX = 8192

# ---------------------------------------------------------------------------------------------
# The fixture: a project shaped like a project, formatting with % throughout.
# ---------------------------------------------------------------------------------------------

FIXTURE: dict[str, str] = {
    "shop/__init__.py": '''\
"""A small storefront, written before f-strings existed."""

__all__ = ["catalogue", "checkout", "money", "orders", "payments"]
''',
    "shop/errors.py": '''\
"""Errors the shop raises, with messages the support team reads."""


class ShopError(Exception):
    """Base class for everything this package raises."""


class OutOfStock(ShopError):
    def __init__(self, sku: str, wanted: int, held: int) -> None:
        super().__init__("sku %s: wanted %d, only %d in stock" % (sku, wanted, held))
        self.sku = sku
        self.wanted = wanted
        self.held = held


class PaymentDeclined(ShopError):
    def __init__(self, reference: str, reason: str) -> None:
        super().__init__("payment %s declined: %s" % (reference, reason))
        self.reference = reference
        self.reason = reason


class UnknownProduct(ShopError):
    def __init__(self, sku: str) -> None:
        super().__init__("no product with sku %s" % sku)
        self.sku = sku


def describe(error: ShopError) -> str:
    """One line for the operations log."""
    return "%s: %s" % (type(error).__name__, error)
''',
    "shop/money.py": '''\
"""Money, in minor units, because floats do not add up."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Money:
    minor: int
    currency: str = "GBP"

    def __str__(self) -> str:
        return "%s%.2f" % (symbol(self.currency), self.minor / 100)

    def __add__(self, other: "Money") -> "Money":
        if other.currency != self.currency:
            raise ValueError("cannot add %s to %s" % (other.currency, self.currency))
        return Money(self.minor + other.minor, self.currency)

    def times(self, quantity: int) -> "Money":
        return Money(self.minor * quantity, self.currency)


def symbol(currency: str) -> str:
    return {"GBP": "\\u00a3", "USD": "$", "EUR": "\\u20ac"}.get(currency, "%s " % currency)


def total(amounts: list[Money]) -> Money:
    if not amounts:
        return Money(0)
    running = amounts[0]
    for amount in amounts[1:]:
        running = running + amount
    return running


def audit_line(label: str, amount: Money) -> str:
    return "%-20s %10s" % (label, str(amount))
''',
    "shop/catalogue.py": '''\
"""The product catalogue, held in memory because this shop is small."""

from shop.errors import UnknownProduct
from shop.money import Money


class Product:
    def __init__(self, sku: str, name: str, price: Money, stock: int) -> None:
        self.sku = sku
        self.name = name
        self.price = price
        self.stock = stock

    def __repr__(self) -> str:
        return "<Product %s %r at %s>" % (self.sku, self.name, self.price)

    def label(self) -> str:
        return "%s (%s) - %s" % (self.name, self.sku, self.price)


class Catalogue:
    def __init__(self, products: list[Product]) -> None:
        self._by_sku = {product.sku: product for product in products}

    def get(self, sku: str) -> Product:
        try:
            return self._by_sku[sku]
        except KeyError:
            raise UnknownProduct(sku) from None

    def summary(self) -> str:
        lines = ["%d product(s)" % len(self._by_sku)]
        for product in self._by_sku.values():
            lines.append("  %s x%d" % (product.label(), product.stock))
        return "\\n".join(lines)
''',
    "shop/orders.py": '''\
"""Orders: lines, quantities and the arithmetic over them."""

from shop.catalogue import Catalogue
from shop.errors import OutOfStock
from shop.money import Money, total


class Line:
    def __init__(self, sku: str, quantity: int, unit: Money) -> None:
        self.sku = sku
        self.quantity = quantity
        self.unit = unit

    def amount(self) -> Money:
        return self.unit.times(self.quantity)

    def __str__(self) -> str:
        return "%d x %s @ %s = %s" % (self.quantity, self.sku, self.unit, self.amount())


class Order:
    def __init__(self, reference: str) -> None:
        self.reference = reference
        self.lines: list[Line] = []

    def add(self, catalogue: Catalogue, sku: str, quantity: int) -> None:
        product = catalogue.get(sku)
        if product.stock < quantity:
            raise OutOfStock(sku, quantity, product.stock)
        self.lines.append(Line(sku, quantity, product.price))

    def subtotal(self) -> Money:
        return total([line.amount() for line in self.lines])

    def describe(self) -> str:
        header = "order %s, %d line(s)" % (self.reference, len(self.lines))
        body = ["  %s" % line for line in self.lines]
        return "\\n".join([header] + body + ["  subtotal %s" % self.subtotal()])
''',
    "shop/payments.py": '''\
"""Taking payment, or failing to."""

from shop.errors import PaymentDeclined
from shop.money import Money


class Gateway:
    def __init__(self, name: str, floor_limit: Money) -> None:
        self.name = name
        self.floor_limit = floor_limit

    def __str__(self) -> str:
        return "%s (floor limit %s)" % (self.name, self.floor_limit)


class Receipt:
    def __init__(self, reference: str, amount: Money, gateway: str) -> None:
        self.reference = reference
        self.amount = amount
        self.gateway = gateway

    def render(self) -> str:
        return "receipt %s: %s via %s" % (self.reference, self.amount, self.gateway)


def charge(gateway: Gateway, reference: str, amount: Money) -> Receipt:
    if amount.minor <= 0:
        raise PaymentDeclined(reference, "amount %s is not positive" % amount)
    if amount.minor > gateway.floor_limit.minor:
        raise PaymentDeclined(
            reference, "amount %s over floor limit %s" % (amount, gateway.floor_limit)
        )
    return Receipt(reference, amount, gateway.name)
''',
    "shop/checkout.py": '''\
"""Checkout: the one place that talks to everything else."""

from shop.money import Money
from shop.orders import Order
from shop.payments import Gateway, Receipt, charge
from shop.tax.calculator import tax_for
from shop.shipping.quotes import cheapest


def checkout(order: Order, gateway: Gateway, country: str, grams: int) -> Receipt:
    goods = order.subtotal()
    tax = tax_for(goods, country)
    postage = cheapest(country, grams)
    payable = goods + tax + postage.price
    return charge(gateway, order.reference, payable)


def quote_lines(order: Order, country: str, grams: int) -> list[str]:
    goods = order.subtotal()
    tax = tax_for(goods, country)
    postage = cheapest(country, grams)
    return [
        "%-12s %s" % ("goods", goods),
        "%-12s %s" % ("tax", tax),
        "%-12s %s (%s)" % ("postage", postage.price, postage.carrier),
        "%-12s %s" % ("payable", Money(goods.minor + tax.minor + postage.price.minor)),
    ]
''',
    "shop/legacy_invoice.py": '''\
"""The invoice renderer nobody has touched since 2014."""

from shop.money import Money
from shop.orders import Order

RULE = "-" * 48


def render(order: Order, country: str, tax: Money, postage: Money) -> str:
    out = ["INVOICE %s" % order.reference, RULE]
    for line in order.lines:
        out.append("%-6s %3d %10s %12s" % (line.sku, line.quantity, line.unit, line.amount()))
    out.append(RULE)
    out.append("%-21s %12s" % ("subtotal", order.subtotal()))
    out.append("%-21s %12s" % ("postage", postage))
    out.append("%-21s %12s" % ("tax (%s)" % country, tax))
    payable = order.subtotal() + postage + tax
    out.append("%-21s %12s" % ("payable", payable))
    out.append(RULE)
    out.append("tax is %.2f%% of goods" % (100.0 * tax.minor / max(order.subtotal().minor, 1)))
    return "\\n".join(out)


def footer(reference: str, terms_days: int) -> str:
    return "%s - payable within %d days - queries quote %s" % (RULE, terms_days, reference)
''',
    "shop/shipping/__init__.py": '"""Getting the parcel to the customer."""\n',
    "shop/shipping/carriers.py": '''\
"""Carriers and what they charge."""

from shop.money import Money


class Carrier:
    def __init__(self, name: str, base: Money, per_kg: Money, countries: list[str]) -> None:
        self.name = name
        self.base = base
        self.per_kg = per_kg
        self.countries = countries

    def serves(self, country: str) -> bool:
        return country in self.countries

    def price(self, grams: int) -> Money:
        kilos = max(1, (grams + 999) // 1000)
        return Money(self.base.minor + self.per_kg.minor * kilos)

    def __str__(self) -> str:
        return "%s: %s + %s/kg to %s" % (
            self.name,
            self.base,
            self.per_kg,
            ", ".join(self.countries),
        )


CARRIERS = [
    Carrier("Royal Mail", Money(299), Money(120), ["GB"]),
    Carrier("DPD", Money(650), Money(240), ["GB", "IE", "FR", "DE"]),
    Carrier("DHL", Money(1150), Money(310), ["GB", "IE", "FR", "DE", "US"]),
]
''',
    "shop/shipping/parcels.py": '''\
"""Parcels: what is in the box and how heavy it is."""


class Parcel:
    def __init__(self, reference: str, grams: int, length_mm: int, girth_mm: int) -> None:
        self.reference = reference
        self.grams = grams
        self.length_mm = length_mm
        self.girth_mm = girth_mm

    def oversized(self) -> bool:
        return self.length_mm + self.girth_mm > 3000

    def __str__(self) -> str:
        return "parcel %s: %dg, %dmm + %dmm girth%s" % (
            self.reference,
            self.grams,
            self.length_mm,
            self.girth_mm,
            " (oversized)" if self.oversized() else "",
        )


def manifest(parcels: list[Parcel]) -> str:
    lines = ["%d parcel(s), %dg total" % (len(parcels), sum(p.grams for p in parcels))]
    for parcel in parcels:
        lines.append("  %s" % parcel)
    return "\\n".join(lines)
''',
    "shop/shipping/quotes.py": '''\
"""Choosing a carrier for a country and a weight."""

from shop.shipping.carriers import CARRIERS, Carrier
from shop.money import Money


class Quote:
    def __init__(self, carrier: str, price: Money) -> None:
        self.carrier = carrier
        self.price = price

    def __str__(self) -> str:
        return "%s for %s" % (self.carrier, self.price)


def cheapest(country: str, grams: int) -> Quote:
    options = [carrier for carrier in CARRIERS if carrier.serves(country)]
    if not options:
        raise ValueError("no carrier serves %s" % country)
    best: Carrier = min(options, key=lambda carrier: carrier.price(grams).minor)
    return Quote(best.name, best.price(grams))


def compare(country: str, grams: int) -> str:
    lines = ["quotes for %s at %dg" % (country, grams)]
    for carrier in CARRIERS:
        if carrier.serves(country):
            lines.append("  %-12s %s" % (carrier.name, carrier.price(grams)))
    return "\\n".join(lines)
''',
    "shop/tax/__init__.py": '"""Tax, by country."""\n',
    "shop/tax/rates.py": '''\
"""Rates, as percentages, because that is how they are published."""

RATES = {"GB": 20.0, "IE": 23.0, "FR": 20.0, "DE": 19.0, "US": 0.0}


def rate_for(country: str) -> float:
    if country not in RATES:
        raise KeyError("no rate published for %s" % country)
    return RATES[country]


def describe(country: str) -> str:
    return "%s: %.2f%%" % (country, rate_for(country))


def table() -> str:
    lines = ["country  rate"]
    for country in sorted(RATES):
        lines.append("%-8s %.2f%%" % (country, RATES[country]))
    return "\\n".join(lines)
''',
    "shop/tax/calculator.py": '''\
"""Applying a published rate to an amount."""

from shop.money import Money
from shop.tax.rates import rate_for


def tax_for(goods: Money, country: str) -> Money:
    rate = rate_for(country)
    return Money(int(round(goods.minor * rate / 100.0)), goods.currency)


def breakdown(goods: Money, country: str) -> str:
    rate = rate_for(country)
    tax = tax_for(goods, country)
    return "%s at %.2f%% = %s (gross %s)" % (goods, rate, tax, goods + tax)
''',
    "shop/discounts.py": '''\
"""Discounts, vouchers and the rules that stop them stacking."""

from shop.money import Money


class Voucher:
    def __init__(self, code: str, percent: float, minimum: Money, stacks: bool) -> None:
        self.code = code
        self.percent = percent
        self.minimum = minimum
        self.stacks = stacks

    def applies_to(self, goods: Money) -> bool:
        return goods.minor >= self.minimum.minor

    def deduction(self, goods: Money) -> Money:
        if not self.applies_to(goods):
            return Money(0, goods.currency)
        return Money(int(round(goods.minor * self.percent / 100.0)), goods.currency)

    def __str__(self) -> str:
        return "%s: %.1f%% off orders over %s%s" % (
            self.code,
            self.percent,
            self.minimum,
            "" if self.stacks else " (does not stack)",
        )


VOUCHERS = [
    Voucher("WELCOME10", 10.0, Money(2000), False),
    Voucher("SPRING5", 5.0, Money(0), True),
    Voucher("BULK15", 15.0, Money(10000), False),
]


def best_for(goods: Money) -> Voucher | None:
    eligible = [voucher for voucher in VOUCHERS if voucher.applies_to(goods)]
    if not eligible:
        return None
    return max(eligible, key=lambda voucher: voucher.deduction(goods).minor)


def explain(goods: Money) -> str:
    lines = ["vouchers against %s" % goods]
    for voucher in VOUCHERS:
        if voucher.applies_to(goods):
            lines.append("  %-10s -%s" % (voucher.code, voucher.deduction(goods)))
        else:
            lines.append("  %-10s not applicable (minimum %s)" % (voucher.code, voucher.minimum))
    winner = best_for(goods)
    lines.append("  best: %s" % (winner.code if winner else "none"))
    return "\\n".join(lines)
''',
    "shop/inventory.py": '''\
"""Stock levels, reservations and the reorder point."""

from shop.errors import OutOfStock


class Stock:
    def __init__(self, sku: str, on_hand: int, reserved: int, reorder_at: int) -> None:
        self.sku = sku
        self.on_hand = on_hand
        self.reserved = reserved
        self.reorder_at = reorder_at

    def available(self) -> int:
        return self.on_hand - self.reserved

    def below_reorder(self) -> bool:
        return self.available() <= self.reorder_at

    def reserve(self, quantity: int) -> None:
        if quantity > self.available():
            raise OutOfStock(self.sku, quantity, self.available())
        self.reserved += quantity

    def release(self, quantity: int) -> None:
        self.reserved = max(0, self.reserved - quantity)

    def __str__(self) -> str:
        return "%-6s on hand %4d, reserved %4d, available %4d%s" % (
            self.sku,
            self.on_hand,
            self.reserved,
            self.available(),
            " REORDER" if self.below_reorder() else "",
        )


class Warehouse:
    def __init__(self, name: str, stock: list[Stock]) -> None:
        self.name = name
        self._by_sku = {item.sku: item for item in stock}

    def get(self, sku: str) -> Stock:
        return self._by_sku[sku]

    def reorder_list(self) -> list[str]:
        return ["%s needs %d more" % (item.sku, item.reorder_at - item.available() + 1)
                for item in self._by_sku.values() if item.below_reorder()]

    def report(self) -> str:
        lines = ["warehouse %s, %d line(s)" % (self.name, len(self._by_sku))]
        for item in self._by_sku.values():
            lines.append("  %s" % item)
        for line in self.reorder_list():
            lines.append("  ! %s" % line)
        return "\\n".join(lines)
''',
    "shop/customers.py": '''\
"""Customers and their addresses."""


class Address:
    def __init__(self, line1: str, town: str, postcode: str, country: str) -> None:
        self.line1 = line1
        self.town = town
        self.postcode = postcode
        self.country = country

    def one_line(self) -> str:
        return "%s, %s, %s, %s" % (self.line1, self.town, self.postcode, self.country)

    def label(self) -> str:
        return "%s\\n%s\\n%s\\n%s" % (self.line1, self.town, self.postcode, self.country)


class Customer:
    def __init__(self, reference: str, name: str, email: str, address: Address) -> None:
        self.reference = reference
        self.name = name
        self.email = email
        self.address = address

    def __str__(self) -> str:
        return "%s <%s> [%s]" % (self.name, self.email, self.reference)

    def greeting(self) -> str:
        first = self.name.split(" ")[0]
        return "Hello %s," % first

    def ships_to(self) -> str:
        return self.address.country


def directory(customers: list[Customer]) -> str:
    lines = ["%d customer(s)" % len(customers)]
    for customer in sorted(customers, key=lambda c: c.name):
        lines.append("  %-28s %-24s %s" % (customer.name, customer.email, customer.ships_to()))
    return "\\n".join(lines)
''',
    "shop/reporting.py": '''\
"""The numbers the owner looks at on a Monday."""

from shop.money import Money


class Row:
    def __init__(self, label: str, orders: int, revenue: Money) -> None:
        self.label = label
        self.orders = orders
        self.revenue = revenue

    def average(self) -> Money:
        if self.orders == 0:
            return Money(0, self.revenue.currency)
        return Money(self.revenue.minor // self.orders, self.revenue.currency)

    def __str__(self) -> str:
        return "%-12s %5d %12s %12s" % (self.label, self.orders, self.revenue, self.average())


def share(row: Row, total_revenue: Money) -> str:
    if total_revenue.minor == 0:
        return "%s: no revenue" % row.label
    return "%s: %.1f%% of revenue" % (row.label, 100.0 * row.revenue.minor / total_revenue.minor)


def table(rows: list[Row]) -> str:
    total = Money(sum(row.revenue.minor for row in rows))
    out = ["%-12s %5s %12s %12s" % ("period", "count", "revenue", "average"), "-" * 44]
    for row in rows:
        out.append(str(row))
    out.append("-" * 44)
    out.append("%-12s %5d %12s" % ("total", sum(row.orders for row in rows), total))
    for row in rows:
        out.append("  %s" % share(row, total))
    return "\\n".join(out)
''',
    "shop/csv_export.py": '''\
"""Exporting orders for the accountant, who wants CSV and nothing else."""

from shop.money import Money
from shop.orders import Order

HEADER = "reference,sku,quantity,unit_minor,amount_minor,amount"


def quote(value: str) -> str:
    if "," in value or '"' in value:
        return '"%s"' % value.replace('"', '""')
    return value


def order_rows(order: Order) -> list[str]:
    rows = []
    for line in order.lines:
        rows.append(
            "%s,%s,%d,%d,%d,%s"
            % (
                quote(order.reference),
                quote(line.sku),
                line.quantity,
                line.unit.minor,
                line.amount().minor,
                line.amount(),
            )
        )
    return rows


def export(orders: list[Order]) -> str:
    out = [HEADER]
    for order in orders:
        out.extend(order_rows(order))
    total = Money(sum(line.amount().minor for order in orders for line in order.lines))
    out.append("# %d order(s), %d line(s), total %s" % (
        len(orders),
        sum(len(order.lines) for order in orders),
        total,
    ))
    return "\\n".join(out)
''',
    "shop/shipping/tracking.py": '''\
"""Tracking events, as the carriers report them."""


class Event:
    def __init__(self, at: str, place: str, status: str) -> None:
        self.at = at
        self.place = place
        self.status = status

    def __str__(self) -> str:
        return "%-18s %-16s %s" % (self.at, self.place, self.status)


class Tracking:
    def __init__(self, reference: str, carrier: str) -> None:
        self.reference = reference
        self.carrier = carrier
        self.events: list[Event] = []

    def add(self, at: str, place: str, status: str) -> None:
        self.events.append(Event(at, place, status))

    def latest(self) -> str:
        if not self.events:
            return "no events for %s" % self.reference
        return "%s: %s" % (self.reference, self.events[-1].status)

    def delivered(self) -> bool:
        return any(event.status.lower() == "delivered" for event in self.events)

    def history(self) -> str:
        head = "tracking %s with %s (%d event(s))" % (
            self.reference,
            self.carrier,
            len(self.events),
        )
        return "\\n".join([head] + ["  %s" % event for event in self.events])
''',
    "shop/tax/exemptions.py": '''\
"""Exemptions, which are the reason the tax module is not one line."""

from shop.money import Money

EXEMPT_CATEGORIES = {"books", "childrens_clothing", "basic_food"}
REDUCED = {"energy": 5.0}


class Exemption:
    def __init__(self, category: str, reason: str) -> None:
        self.category = category
        self.reason = reason

    def __str__(self) -> str:
        return "%s is exempt (%s)" % (self.category, self.reason)


def exempt(category: str) -> bool:
    return category in EXEMPT_CATEGORIES


def effective_rate(category: str, standard: float) -> float:
    if exempt(category):
        return 0.0
    return REDUCED.get(category, standard)


def explain(category: str, standard: float, goods: Money) -> str:
    rate = effective_rate(category, standard)
    if rate == 0.0:
        return "%s: exempt, %s taxed at %.2f%%" % (category, goods, rate)
    if category in REDUCED:
        return "%s: reduced rate %.2f%% instead of %.2f%%" % (category, rate, standard)
    return "%s: standard rate %.2f%%" % (category, rate)
''',
    "shop/notifications/__init__.py": '"""Telling the customer what happened."""\n',
    "shop/notifications/digest.py": '''\
"""The nightly digest nobody reads until something goes wrong."""

from shop.money import Money


class Entry:
    def __init__(self, kind: str, reference: str, detail: str) -> None:
        self.kind = kind
        self.reference = reference
        self.detail = detail

    def __str__(self) -> str:
        return "[%-8s] %-10s %s" % (self.kind, self.reference, self.detail)


class Digest:
    def __init__(self, day: str) -> None:
        self.day = day
        self.entries: list[Entry] = []

    def add(self, kind: str, reference: str, detail: str) -> None:
        self.entries.append(Entry(kind, reference, detail))

    def counts(self) -> dict[str, int]:
        counted: dict[str, int] = {}
        for entry in self.entries:
            counted[entry.kind] = counted.get(entry.kind, 0) + 1
        return counted

    def render(self, takings: Money) -> str:
        head = "digest for %s - %d entry(ies), takings %s" % (
            self.day,
            len(self.entries),
            takings,
        )
        body = ["  %s" % entry for entry in self.entries]
        tail = ["  %-10s %d" % (kind, count) for kind, count in sorted(self.counts().items())]
        return "\\n".join([head] + body + ["  --"] + tail)
''',
    "shop/notifications/templates.py": '''\
"""Message templates. Marketing owns the wording; do not edit without them."""

ORDER_PLACED = "Thanks! Order %s is confirmed. Total %s."
ORDER_SHIPPED = "Order %s is on its way with %s. Expect it in %d day(s)."
PAYMENT_FAILED = "We could not take payment for order %s: %s."


def order_placed(reference: str, total: str) -> str:
    return ORDER_PLACED % (reference, total)


def order_shipped(reference: str, carrier: str, days: int) -> str:
    return ORDER_SHIPPED % (reference, carrier, days)


def payment_failed(reference: str, reason: str) -> str:
    return PAYMENT_FAILED % (reference, reason)
''',
    "shop/notifications/dispatch.py": '''\
"""Sending the message, or recording that we could not."""

from shop.notifications import templates

SENT: list[str] = []


def send(address: str, body: str) -> str:
    if "@" not in address:
        raise ValueError("not an address: %s" % address)
    record = "to %s: %s" % (address, body)
    SENT.append(record)
    return record


def confirm(address: str, reference: str, total: str) -> str:
    return send(address, templates.order_placed(reference, total))


def shipped(address: str, reference: str, carrier: str, days: int) -> str:
    return send(address, templates.order_shipped(reference, carrier, days))


def log() -> str:
    return "\\n".join("%3d %s" % (i, line) for i, line in enumerate(SENT, 1))
''',
    "tests/test_shop.py": '''\
from shop.catalogue import Catalogue, Product
from shop.legacy_invoice import render
from shop.money import Money, total
from shop.notifications import dispatch
from shop.orders import Order
from shop.payments import Gateway, charge
from shop.shipping.carriers import CARRIERS
from shop.shipping.parcels import Parcel
from shop.shipping.quotes import cheapest
from shop.tax.calculator import tax_for
from shop.tax.rates import rate_for


def catalogue() -> Catalogue:
    return Catalogue(
        [
            Product("A1", "Kettle", Money(2499), 4),
            Product("B2", "Toaster", Money(1899), 2),
        ]
    )


def order() -> Order:
    ordered = Order("ORD-1")
    ordered.add(catalogue(), "A1", 2)
    ordered.add(catalogue(), "B2", 1)
    return ordered


def test_money_formats_to_two_places():
    assert str(Money(2499)) == "\\u00a324.99"


def test_money_adds():
    assert total([Money(100), Money(250)]).minor == 350


def test_product_label():
    assert catalogue().get("A1").label() == "Kettle (A1) - \\u00a324.99"


def test_order_subtotal():
    assert order().subtotal().minor == 2499 * 2 + 1899


def test_order_describes_itself():
    assert "order ORD-1, 2 line(s)" in order().describe()


def test_tax_is_applied():
    assert tax_for(Money(1000), "GB").minor == 200


def test_rate_lookup():
    assert rate_for("DE") == 19.0


def test_cheapest_carrier_for_gb():
    assert cheapest("GB", 500).carrier == "Royal Mail"


def test_every_carrier_prices_a_kilo():
    assert all(carrier.price(1000).minor > 0 for carrier in CARRIERS)


def test_parcel_oversize():
    assert Parcel("P1", 900, 2000, 1500).oversized()


def test_charge_produces_a_receipt():
    receipt = charge(Gateway("stripe", Money(50000)), "ORD-1", Money(1000))
    assert "receipt ORD-1" in receipt.render()


def test_invoice_renders():
    text = render(order(), "GB", Money(1379), Money(299))
    assert "INVOICE ORD-1" in text


def test_dispatch_records_what_it_sent():
    dispatch.SENT.clear()
    dispatch.confirm("a@b.com", "ORD-1", "\\u00a368.97")
    assert len(dispatch.SENT) == 1
''',
    "pyproject.toml": """\
[project]
name = "shop"
version = "0.1.0"
requires-python = ">=3.12"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
""",
}

PROMPT = """\
# Modernise every string in `shop/` to f-strings

The package predates f-strings and formats with the `%` operator throughout. Convert all of it,
file by file, without changing behaviour or the text any user sees.

## 1. Scope

Convert every `%`-format expression in each of these, and nothing else:

- `shop/errors.py` - the exception messages the support team reads
- `shop/money.py` - `__str__`, `symbol`, `audit_line`
- `shop/catalogue.py` - `Product.__repr__`, `Product.label`, `Catalogue.summary`
- `shop/orders.py` - `Line.__str__`, `Order.describe`
- `shop/payments.py` - `Gateway.__str__`, `Receipt.render`, the two decline messages
- `shop/checkout.py` - `quote_lines`
- `shop/discounts.py` - `Voucher.__str__` and `explain`
- `shop/inventory.py` - `Stock.__str__`, `Warehouse.reorder_list`, `Warehouse.report`
- `shop/customers.py` - `Address.one_line`, `Address.label`, `Customer.__str__`, `directory`
- `shop/reporting.py` - `Row.__str__`, `share`, `table`
- `shop/csv_export.py` - `quote`, `order_rows`, `export`
- `shop/legacy_invoice.py` - `render` and `footer`
- `shop/shipping/carriers.py` - `Carrier.__str__`
- `shop/shipping/parcels.py` - `Parcel.__str__`, `manifest`
- `shop/shipping/quotes.py` - `Quote.__str__`, `cheapest`, `compare`
- `shop/shipping/tracking.py` - `Event.__str__`, `Tracking.latest`, `Tracking.history`
- `shop/tax/rates.py` - `rate_for`, `describe`, `table`
- `shop/tax/calculator.py` - `breakdown`
- `shop/tax/exemptions.py` - `Exemption.__str__` and `explain`
- `shop/notifications/dispatch.py` - `send`, `log`
- `shop/notifications/digest.py` - `Entry.__str__` and `Digest.render`

Do not touch `shop/notifications/templates.py`: those constants are `%`-templates applied at
call time and marketing owns the wording. Do not touch the tests.

## 2. Rules

1. Every converted expression is an f-string. No percent operator, no `str` formatting method,
   no string concatenation.
2. The rendered output must be byte-for-byte what it was. The tests assert on it.
3. Width and precision specifiers carry over into the format spec: `%-20s` becomes `:<20`,
   `%10s` becomes `:>10`, `%.2f` becomes `:.2f`, `%3d` becomes `:3d`, `%03d` becomes `:03d`.
4. A literal percent sign is `%%` in a `%`-format string and a single `%` in an f-string.
   `shop/tax/rates.py` and `shop/legacy_invoice.py` both publish percentages this way and both
   are easy to get wrong.
5. Do not rename anything, do not reorder functions, do not add or remove behaviour, do not
   introduce a helper to do the formatting.
6. Keep every line within 100 columns; the project's ruff config enforces it.

## 3. Worked examples

Before:

    return "sku %s: wanted %d, only %d in stock" % (sku, wanted, held)
    return "%-20s %10s" % (label, str(amount))
    return "%s: %.2f%%" % (country, rate_for(country))

After:

    return f"sku {sku}: wanted {wanted:d}, only {held:d} in stock"
    return f"{label:<20} {str(amount):>10}"
    return f"{country}: {rate_for(country):.2f}%"

## 4. Order of work

Start at the leaves and work up, so a mistake surfaces in a small file first: `errors.py`,
`money.py`, then the shipping and tax packages, then `catalogue.py`, `orders.py`,
`payments.py`, `checkout.py`, and `legacy_invoice.py` last because it is the longest.

## 5. Definition of done

- `python -m ast` parses every file that was changed.
- `ruff check .` is clean.
- `pytest -q` passes all thirteen tests, unchanged.
- `grep -rn "%" shop/` returns hits only in `shop/notifications/templates.py`.
"""


def build_fixture(root: Path, config_source: Path) -> None:
    """Write the project, its config and a git repository holding a clean baseline."""
    for relative, content in FIXTURE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "prompt.txt").write_text(PROMPT, encoding="utf-8")

    # The backend, model and tokenizer come from this machine's own config, so the shots are of
    # the setup the reader would have. Everything else is the fixture's.
    inherited = []
    if config_source.exists():
        for line in config_source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("backend_url", "model", "gguf_path", "kv_mib_per_1k")):
                inherited.append(stripped)
    (root / "pharos.toml").write_text(
        "\n".join(
            [
                "# Written by docs/capture_cli.py. num_ctx is deliberately small: the prompt",
                "# not fitting is the thing being photographed.",
                *inherited,
                'target_folder = "."',
                f"num_ctx = {SHOT_NUM_CTX}",
                "log_file = \"pharos.log\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "pharos.toml\npharos.log\npharos_*.json\n.pharos/\n", encoding="utf-8"
    )

    git = ["git", "-c", "user.email=capture@pharos.local", "-c", "user.name=Pharos Capture"]
    subprocess.run([*git, "init", "-q"], cwd=root, check=True)
    subprocess.run([*git, "add", "-A"], cwd=root, check=True)
    subprocess.run([*git, "commit", "-qm", "shop, before the refactor"], cwd=root, check=True)


def warm_backend(root: Path) -> None:
    """Load the model at the shot's window, so the shots carry a verdict instead of N/A.

    `pharos check` reports NO VERDICT when nothing is resident — correct, and a poor
    photograph. One generation of a single token is the cheapest way to make the budget in the
    picture the budget a reader would actually have.
    """
    import httpx

    from pharos.config import load_config

    previous = os.getcwd()
    os.chdir(root)
    try:
        config = load_config()
    finally:
        os.chdir(previous)
    if config.model is None:
        print("no model configured — shots will report NO VERDICT", file=sys.stderr)
        return
    print(f"loading {config.model} at num_ctx {SHOT_NUM_CTX} …", file=sys.stderr)
    try:
        response = httpx.post(
            f"{config.backend_url.rstrip('/')}/api/chat",
            json={
                "model": config.model,
                "stream": False,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "options": {"num_predict": 1, "num_ctx": SHOT_NUM_CTX},
            },
            timeout=300.0,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — a shot without a backend is worth taking anyway
        print(f"backend did not answer ({exc}) — shots will report what that looks like",
              file=sys.stderr)


def run_cli(args: list[str], cwd: Path) -> tuple[str, str, int]:
    """Run the real CLI in a child process that Rich will write ANSI to."""
    boot = (
        "import rich.console as rc;"
        "rc.detect_legacy_windows=lambda: False;"
        "import sys;sys.argv=['pharos']+sys.argv[1:];"
        "from pharos.__main__ import main;main()"
    )
    env = dict(
        os.environ,
        FORCE_COLOR="1",
        COLORTERM="truecolor",
        TERM="xterm-256color",
        COLUMNS=str(WIDTH + 1),  # Rich takes one column off a Windows console
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(REPO),
    )
    completed = subprocess.run(
        [sys.executable, "-c", boot, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # The two streams are kept apart on purpose. `pharos run` writes its report to stdout and
    # its live progress — a spinner, redrawn several times a second — to stderr, and a capture
    # that merges them photographs the spinner's last frame instead of the scorecard.
    return completed.stdout.rstrip("\n"), completed.stderr.rstrip("\n"), completed.returncode


_PROGRESS_BANNER = (
    "===== everything below is the live progress written to stderr while the above was "
    "being produced ====="
)


def _plain(line: str) -> str:
    """The line as a reader sees it, with the ANSI escapes taken back out."""
    out: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\x1b" and line[index + 1 : index + 2] == "[":
            end = index + 2
            while end < len(line) and not line[end].isalpha():
                end += 1
            index = end + 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def save_svg(
    captured: str,
    out: Path,
    title: str,
    tail: int | None = None,
    region: tuple[str, str] | None = None,
) -> None:
    """Render captured terminal output to an SVG.

    A whole run is hundreds of lines and an SVG of it is unreadable, so a shot may be trimmed —
    either to the last `tail` lines or to a `region` between two markers. Both say so in the
    window title, and the full transcript is written beside the image either way: a screenshot
    that quietly drops the part where it went wrong is the one thing this must not be.
    """
    lines = captured.splitlines()
    if region is not None:
        start_marker, end_marker = region
        starts = [i for i, line in enumerate(lines) if _plain(line).startswith(start_marker)]
        # The first end marker AFTER the start, not the last in the file: a run prints two
        # panels that close identically, and the region is meant to take one of them.
        ends = (
            [i for i, line in enumerate(lines) if _plain(line).startswith(end_marker)]
            if starts
            else []
        )
        after = [i for i in ends if i > starts[0]] if starts else []
        if after:
            lines = lines[starts[0] : after[0] + 1]
            title = f"{title}  (excerpt)"
        else:
            print(f"  region {region!r} not found — falling back to the tail", file=sys.stderr)
    if tail is not None and len(lines) > tail:
        lines = lines[-tail:]
        title = f"{title}  (last {tail} lines)"
    console = Console(record=True, width=WIDTH, file=io.StringIO(), legacy_windows=False)
    console.print(Text.from_ansi("\n".join(lines)))
    console.save_svg(str(out), title=title, theme=MONOKAI)
    print(f"wrote {out.relative_to(REPO)} ({len(lines)} lines)", file=sys.stderr)


class Image:
    """One SVG cut out of one capture."""

    def __init__(
        self,
        filename: str,
        title: str,
        tail: int | None = None,
        region: tuple[str, str] | None = None,
    ) -> None:
        self.filename = filename
        self.title = title
        self.tail = tail
        self.region = region


class Shot:
    """One command, and the images cut from what it printed."""

    def __init__(self, argv: list[str], transcript: str, images: list[Image]) -> None:
        self.argv = argv
        self.transcript = transcript
        self.images = images


SHOTS: dict[str, Shot] = {
    "check": Shot(
        ["check", "--file", "prompt.txt"],
        "shot-check.txt",
        [Image("shot-check.svg", "pharos check")],
    ),
    "split": Shot(
        ["split", "--file", "prompt.txt", "--semantic"],
        "shot-split.txt",
        [
            Image(
                "shot-split.svg",
                "pharos split --semantic",
                region=("FLOOR", "Each projection is a floor"),
            )
        ],
    ),
    "run": Shot(
        [
            "run",
            "--file",
            "prompt.txt",
            "--semantic",
            "--compact",
            "--review",
            "--exclude",
            "shop/notifications/templates.py",
        ],
        "shot-run.txt",
        [
            # Two images out of one run: what it did and how it scored, then the opinion that
            # is printed under a verdict it could not change. Together they are 90 lines and
            # unreadable as one picture.
            Image(
                "shot-run.svg",
                "pharos run --semantic --compact --review",
                region=("  Pharos run", "  ╰─"),
            ),
            Image(
                "shot-review.svg",
                "…the review, printed last",
                region=("  ╭─ Review", "before it was asked"),
            ),
        ],
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        default="check,split,run",
        help="comma-separated shots to take: check, split, run (default: all three)",
    )
    parser.add_argument("--keep", action="store_true", help="leave the fixture project on disk")
    parser.add_argument(
        "--project",
        type=Path,
        help="capture against this directory instead of building the fixture",
    )
    args = parser.parse_args()

    wanted = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in SHOTS]
    if unknown:
        parser.error(f"unknown shot(s): {', '.join(unknown)}")

    if args.project is not None:
        root = args.project.resolve()
        temporary = False
    else:
        root = Path(tempfile.mkdtemp(prefix="pharos-shots-")).resolve()
        temporary = True
        build_fixture(root, REPO / "pharos.toml")
        print(f"fixture: {root}", file=sys.stderr)

    warm_backend(root)

    try:
        for name in wanted:
            shot = SHOTS[name]
            print(f"capturing {name} …", file=sys.stderr)
            report, progress, code = run_cli(shot.argv, root)
            if not report.strip():
                print(f"  {name} printed nothing (exit {code}) — not written", file=sys.stderr)
                continue
            (DOCS / shot.transcript).write_text(
                f"{report}\n\n{_PROGRESS_BANNER}\n\n{progress}\n", encoding="utf-8"
            )
            for image in shot.images:
                save_svg(
                    report,
                    DOCS / image.filename,
                    f"{image.title}   ·   exit {code}",
                    image.tail,
                    image.region,
                )
    finally:
        if temporary and not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        elif temporary:
            print(f"fixture kept at {root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
