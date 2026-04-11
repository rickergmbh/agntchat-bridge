"""
Google Places photo resolution for ResultPresentation items.

Uses the Google Places API (New) to search for places by name and
return real, working photo URLs that render in the mobile app.

Requires GOOGLE_PLACES_API_KEY environment variable.

Usage:
    from google_places import enrich_presentation_photos

    # Post-process a parsed result_presentation dict
    await enrich_presentation_photos(data, default_lat=50.94, default_lng=6.96)
    # Items now have real image_url and gallery_images values
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger("google_places")

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
# Feature flag: set ENABLE_PHOTO_ENRICHMENT=1 to enable Google Places photo lookups.
# Disabled by default — Text Search (New) costs $32/1,000 requests.
PHOTO_ENRICHMENT_ENABLED = os.getenv("ENABLE_PHOTO_ENRICHMENT", "0") == "1"

# Places API (New) endpoints
_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_PHOTO_MEDIA_URL = "https://places.googleapis.com/v1/{name}/media"

# Max photos to resolve per item (primary + gallery)
_MAX_GALLERY = 4

# In-memory cache: place name -> (photo_url, gallery_urls)
_photo_cache: dict[str, tuple[str | None, list[str]]] = {}


def _is_real_url(url: str | None) -> bool:
    """Check if a URL looks like a real, fetchable image (not hallucinated)."""
    if not url:
        return False
    # Common hallucination patterns
    hallucinated = [
        "example.com",
        "placeholder",
        "dummy",
        "lorem",
        "picsum",
        "via.placeholder",
        "placehold.it",
    ]
    lower = url.lower()
    return not any(p in lower for p in hallucinated)


def _photo_url(photo_name: str, max_height: int = 800) -> str:
    """Build a Places Photos URL that resolves to a real image.

    The URL with key param will redirect to the actual hosted photo
    when loaded by the mobile app's <Image> component.
    """
    return (
        f"{_PHOTO_MEDIA_URL.format(name=photo_name)}"
        f"?maxHeightPx={max_height}&key={GOOGLE_PLACES_API_KEY}"
    )


async def _search_place_photos(
    query: str,
    lat: float | None = None,
    lng: float | None = None,
    max_photos: int = 5,
) -> tuple[str | None, list[str]]:
    """Search for a place and return (primary_url, gallery_urls).

    Uses Places API (New) Text Search with photo field mask.
    """
    if not GOOGLE_PLACES_API_KEY:
        return None, []

    # Check cache
    cache_key = f"{query}|{lat}|{lng}"
    if cache_key in _photo_cache:
        return _photo_cache[cache_key]

    import httpx

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.photos,places.displayName",
    }

    body: dict[str, Any] = {
        "textQuery": query,
        "maxResultCount": 1,
    }

    # Add location bias if coordinates available
    if lat is not None and lng is not None:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 50000.0,  # 50km radius
            }
        }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(_TEXT_SEARCH_URL, json=body, headers=headers)

        if resp.status_code != 200:
            logger.warning("Places API returned %d: %s", resp.status_code, resp.text[:200])
            _photo_cache[cache_key] = (None, [])
            return None, []

        data = resp.json()
        places = data.get("places", [])
        if not places:
            logger.debug("No places found for query: %s", query)
            _photo_cache[cache_key] = (None, [])
            return None, []

        photos = places[0].get("photos", [])
        if not photos:
            logger.debug("Place found but no photos for: %s", query)
            _photo_cache[cache_key] = (None, [])
            return None, []

        # Build URLs: first = primary, rest = gallery
        primary = _photo_url(photos[0]["name"])
        gallery = [_photo_url(p["name"]) for p in photos[1:max_photos]]

        _photo_cache[cache_key] = (primary, gallery)
        logger.info(
            "Resolved %d photo(s) for '%s'",
            1 + len(gallery),
            query,
        )
        return primary, gallery

    except Exception as e:
        logger.warning("Places API error for '%s': %s", query, e)
        _photo_cache[cache_key] = (None, [])
        return None, []


def _build_search_query(item: dict[str, Any]) -> str:
    """Build a search query from an item's title and location context."""
    parts = [item.get("title", "")]

    # Add location from item if available
    loc = item.get("location")
    if isinstance(loc, dict) and loc.get("address"):
        parts.append(loc["address"])
    elif isinstance(loc, str):
        parts.append(loc)
    elif item.get("subtitle"):
        parts.append(item["subtitle"])

    return " ".join(p for p in parts if p)


def _extract_coords(item: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract lat/lng from an item's location field."""
    loc = item.get("location")
    if isinstance(loc, dict):
        lat = loc.get("lat") or loc.get("latitude")
        lng = loc.get("lng") or loc.get("longitude")
        if lat is not None and lng is not None:
            return float(lat), float(lng)
    return None, None


async def enrich_item_photos(
    item: dict[str, Any],
    default_lat: float | None = None,
    default_lng: float | None = None,
) -> None:
    """Enrich a single item dict with real Google Places photos.

    Modifies the item in-place. Skips if image_url is already a real URL.
    """
    # Skip types that don't benefit from photos
    item_type = item.get("type", "generic")
    if item_type == "flight":
        return  # Flights use airline logos, not place photos

    # Already has a real image? Skip
    if _is_real_url(item.get("image_url")):
        return

    query = _build_search_query(item)
    if not query.strip():
        return

    lat, lng = _extract_coords(item)
    lat = lat or default_lat
    lng = lng or default_lng

    primary, gallery = await _search_place_photos(
        query, lat=lat, lng=lng, max_photos=_MAX_GALLERY + 1
    )

    if primary:
        item["image_url"] = primary
    if gallery:
        existing_gallery = item.get("gallery_images") or []
        # Merge: keep any existing real URLs, add new ones
        real_existing = [u for u in existing_gallery if _is_real_url(u)]
        item["gallery_images"] = (real_existing + gallery)[:_MAX_GALLERY]


async def enrich_presentation_photos(
    data: dict[str, Any],
    default_lat: float | None = None,
    default_lng: float | None = None,
) -> None:
    """Enrich all items in a parsed result_presentation dict with real photos.

    Modifies data["items"] in-place. Runs photo lookups concurrently.
    """
    if not PHOTO_ENRICHMENT_ENABLED:
        logger.debug("Photo enrichment disabled (set ENABLE_PHOTO_ENRICHMENT=1 to enable)")
        return

    if not GOOGLE_PLACES_API_KEY:
        logger.debug("GOOGLE_PLACES_API_KEY not set, skipping photo enrichment")
        return

    items = data.get("items", [])
    if not items:
        return

    # Run all lookups concurrently
    await asyncio.gather(
        *(enrich_item_photos(item, default_lat, default_lng) for item in items)
    )
