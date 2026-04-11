"""Typed result models for building ResultPresentation messages.

Provides dataclasses for each result type (hotel, flight, restaurant, event,
product, generic) with validation and serialization to the canonical ACP v2
format that the mobile app expects.

Usage::

    from agentchat.results import (
        ResultPresentation, HotelItem, HotelDetails, Price, CTA, CTABlock, Citation,
        DetailField, DetailTemplate,
    )

    result = ResultPresentation(
        result_type="hotel",
        title="Top Hotels in Berlin",
        items=[
            HotelItem(
                title="Hotel Adlon Kempinski",
                subtitle="Unter den Linden 77, Berlin",
                image_url="https://example.com/hero.jpg",
                gallery_images=["https://example.com/room.jpg", "https://example.com/pool.jpg"],
                rating=4.8,
                rating_count=2341,
                price=Price(amount=285, currency="EUR", per="night"),
                amenities=["Spa", "Pool", "WiFi"],
                details=HotelDetails(
                    check_in="15:00", check_out="11:00",
                    room_type="Deluxe King", cancellation="Free cancellation",
                ).to_dict(),
            ),
        ],
        citations=[Citation(source_name="Booking.com", confidence=0.95)],
    )

    await client.send_result_presentation(conversation_id, result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


@dataclass
class Price:
    """Price with currency and optional discount."""

    amount: float
    currency: str = "USD"
    per: str | None = None
    original_amount: float | None = None
    discount_pct: float | None = None

    def validate(self) -> None:
        if not isinstance(self.amount, (int, float)):
            raise ValueError("Price.amount must be a number")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {"amount": self.amount, "currency": self.currency}
        if self.per is not None:
            d["per"] = self.per
        if self.original_amount is not None:
            d["original_amount"] = self.original_amount
        if self.discount_pct is not None:
            d["discount_pct"] = self.discount_pct
        return d


@dataclass
class CTA:
    """A single call-to-action button."""

    label: str
    url: str | None = None
    action: str | None = None

    def validate(self) -> None:
        if not self.label or not isinstance(self.label, str):
            raise ValueError("CTA.label must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {"label": self.label}
        if self.url is not None:
            d["url"] = self.url
        if self.action is not None:
            d["action"] = self.action
        return d


@dataclass
class CTABlock:
    """Primary + optional secondary CTA buttons."""

    primary: CTA | None = None
    secondary: list[CTA] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.primary is not None:
            d["primary"] = self.primary.to_dict()
        if self.secondary:
            d["secondary"] = [s.to_dict() for s in self.secondary]
        return d


@dataclass
class Citation:
    """Source citation for result data."""

    source_name: str
    source_url: str | None = None
    scraped_at: str | None = None
    confidence: float | None = None

    def validate(self) -> None:
        if not self.source_name or not isinstance(self.source_name, str):
            raise ValueError("Citation.source_name must be a non-empty string")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Citation.confidence must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {"source_name": self.source_name}
        if self.source_url is not None:
            d["source_url"] = self.source_url
        if self.scraped_at is not None:
            d["scraped_at"] = self.scraped_at
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d


@dataclass
class Location:
    """Geographic location."""

    lat: float
    lng: float
    address: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"lat": self.lat, "lng": self.lng}
        if self.address is not None:
            d["address"] = self.address
        return d


# ---------------------------------------------------------------------------
# Detail templates
# ---------------------------------------------------------------------------

VALID_DISPLAYS = {"row", "chip", "highlight"}


@dataclass
class DetailField:
    """A single field descriptor in a detail template.

    Describes how to render one key from the item's ``details`` map.
    """

    key: str
    display: str  # "row" | "chip" | "highlight"
    label: str | None = None
    icon: str | None = None
    color: str | None = None
    format: str | None = None
    hidden: bool = False

    def validate(self) -> None:
        if not self.key or not isinstance(self.key, str):
            raise ValueError("DetailField.key must be a non-empty string")
        if self.display not in VALID_DISPLAYS:
            raise ValueError(
                f"DetailField.display must be one of {VALID_DISPLAYS}, got {self.display!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {"key": self.key, "display": self.display}
        if self.label is not None:
            d["label"] = self.label
        if self.icon is not None:
            d["icon"] = self.icon
        if self.color is not None:
            d["color"] = self.color
        if self.format is not None:
            d["format"] = self.format
        if self.hidden:
            d["hidden"] = True
        return d


@dataclass
class DetailTemplate:
    """A named collection of :class:`DetailField` descriptors.

    Register in ``structured_capabilities.detail_templates`` so the backend
    can resolve them by name at message insertion time.
    """

    name: str
    fields: list[DetailField]

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize to the list format expected by the backend."""
        return [f.to_dict() for f in self.fields]

    def to_capability_entry(self) -> tuple[str, list[dict[str, Any]]]:
        """Return ``(name, schema_list)`` for embedding in structured_capabilities."""
        return (self.name, self.to_list())


# ---------------------------------------------------------------------------
# Base result item
# ---------------------------------------------------------------------------

VALID_RESULT_TYPES = {"hotel", "flight", "restaurant", "event", "product", "generic", "email", "finance", "contact"}


@dataclass
class ResultItem:
    """Base result item. Use typed subclasses for validation."""

    title: str
    type: str = "generic"
    subtitle: str | None = None
    image_url: str | None = None
    rating: float | None = None
    rating_count: int | None = None
    rating_source: str | None = None
    price: Price | None = None
    amenities: list[str] | None = None
    highlights: list[str] | None = None
    booking_url: str | None = None
    cta: CTABlock | None = None
    location: str | Location | None = None
    details: dict[str, Any] | None = None
    detail_template: str | None = None
    detail_schema: list[dict[str, Any]] | None = None

    def validate(self) -> None:
        if not self.title or not isinstance(self.title, str):
            raise ValueError("ResultItem.title must be a non-empty string")
        if self.rating is not None and not (0.0 <= self.rating <= 5.0):
            raise ValueError("ResultItem.rating must be between 0.0 and 5.0")
        if self.price is not None:
            self.price.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {"title": self.title, "type": self.type}
        if self.subtitle is not None:
            d["subtitle"] = self.subtitle
        if self.image_url is not None:
            d["image_url"] = self.image_url
        if self.rating is not None:
            d["rating"] = self.rating
        if self.rating_count is not None:
            d["rating_count"] = self.rating_count
        if self.rating_source is not None:
            d["rating_source"] = self.rating_source
        if self.price is not None:
            d["price"] = self.price.to_dict()
        if self.amenities is not None:
            d["amenities"] = self.amenities
        if self.highlights is not None:
            d["highlights"] = self.highlights
        if self.booking_url is not None:
            d["booking_url"] = self.booking_url
        if self.cta is not None:
            d["cta"] = self.cta.to_dict()
        if self.location is not None:
            d["location"] = (
                self.location.to_dict()
                if isinstance(self.location, Location)
                else self.location
            )
        if self.details is not None:
            d["details"] = self.details
        if self.detail_template is not None:
            d["detail_template"] = self.detail_template
        if self.detail_schema is not None:
            d["detail_schema"] = self.detail_schema
        return d


# ---------------------------------------------------------------------------
# Typed result items
# ---------------------------------------------------------------------------


@dataclass
class HotelDetails:
    """Hotel-specific detail fields."""

    check_in: str | None = None
    check_out: str | None = None
    room_type: str | None = None
    distance: str | None = None
    cancellation: str | None = None
    star_class: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.check_in is not None:
            d["check_in"] = self.check_in
        if self.check_out is not None:
            d["check_out"] = self.check_out
        if self.room_type is not None:
            d["room_type"] = self.room_type
        if self.distance is not None:
            d["distance"] = self.distance
        if self.cancellation is not None:
            d["cancellation"] = self.cancellation
        if self.star_class is not None:
            d["star_class"] = self.star_class
        return d


@dataclass
class HotelItem(ResultItem):
    """Hotel result — requires price."""

    type: str = "hotel"
    gallery_images: list[str] | None = None

    def validate(self) -> None:
        super().validate()
        if self.price is None:
            raise ValueError("HotelItem requires a price")

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.gallery_images:
            d["gallery_images"] = self.gallery_images
        return d


@dataclass
class FlightItem(ResultItem):
    """Flight result — requires departure and arrival in details."""

    type: str = "flight"

    def validate(self) -> None:
        super().validate()
        if not self.details or "departure" not in self.details:
            raise ValueError("FlightItem requires details.departure")
        if "arrival" not in self.details:
            raise ValueError("FlightItem requires details.arrival")


@dataclass
class RestaurantItem(ResultItem):
    """Restaurant result — requires cuisine in details."""

    type: str = "restaurant"

    def validate(self) -> None:
        super().validate()
        if not self.details or "cuisine" not in self.details:
            raise ValueError("RestaurantItem requires details.cuisine")


@dataclass
class EventItem(ResultItem):
    """Event result — requires date and venue in details."""

    type: str = "event"

    def validate(self) -> None:
        super().validate()
        if not self.details or "date" not in self.details:
            raise ValueError("EventItem requires details.date")
        if "venue" not in self.details:
            raise ValueError("EventItem requires details.venue")


@dataclass
class ProductItem(ResultItem):
    """Product result — requires price."""

    type: str = "product"

    def validate(self) -> None:
        super().validate()
        if self.price is None:
            raise ValueError("ProductItem requires a price")


@dataclass
class GenericItem(ResultItem):
    """Generic result — no additional requirements beyond title."""

    type: str = "generic"


# ---------------------------------------------------------------------------
# Top-level presentation
# ---------------------------------------------------------------------------


@dataclass
class ResultPresentation:
    """A complete result presentation with typed items, citations, and metadata.

    Validates on construction via :meth:`validate` and serializes to the
    canonical flat dict format expected by the AgentChat backend and mobile.
    """

    result_type: str
    items: list[ResultItem]
    title: str | None = None
    citations: list[Citation] | None = None
    metadata: dict[str, Any] | None = None
    task_id: str | None = None

    def validate(self) -> None:
        """Validate the entire presentation. Raises ValueError on failure."""
        if self.result_type not in VALID_RESULT_TYPES:
            raise ValueError(
                f"result_type must be one of {VALID_RESULT_TYPES}, got {self.result_type!r}"
            )
        if not self.items:
            raise ValueError("items must be a non-empty list")
        for item in self.items:
            item.validate()
        if self.citations:
            for c in self.citations:
                c.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical flat dict for ACP v2 data payload."""
        self.validate()
        d: dict[str, Any] = {
            "result_type": self.result_type,
            "items": [item.to_dict() for item in self.items],
        }
        if self.title is not None:
            d["title"] = self.title
        if self.citations:
            d["citations"] = [c.to_dict() for c in self.citations]
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.task_id is not None:
            d["task_id"] = self.task_id
        return d
