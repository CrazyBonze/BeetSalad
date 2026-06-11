#!/usr/bin/env python3
"""
mb_debug.py — MusicBrainz Release Group Inspector
Usage:
    python mb_debug.py <release-group-mbid>
    python mb_debug.py 0a59decb-2c3f-3ded-bfda-f7b4a2b09bfd

Prints raw MusicBrainz data for a release group and all its releases,
showing exactly what Lidarr sees when it looks up an album.
"""

import sys
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

MB_API = "https://musicbrainz.org/ws/2"
HEADERS = {
    "User-Agent": "mb-debug-tool/1.0 (beets-lidarr-debugging)",
    "Accept": "application/json",
}


def mb_get(endpoint: str, params: dict) -> dict:
    """Fetch from MusicBrainz API with rate limiting."""
    param_str = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{MB_API}/{endpoint}?{param_str}"
    print(f"  [GET] {url}", flush=True)

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  [HTTP ERROR] {e.code}: {e.reason}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"  [URL ERROR] {e.reason}")
        sys.exit(1)
    finally:
        time.sleep(1.1)  # MusicBrainz rate limit: 1 req/sec



def fmt(label: str, value: Any, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}{label:<30} {value}")


def print_artist_credits(credits: list, indent: int = 0) -> None:
    parts = []
    for credit in credits:
        if isinstance(credit, dict):
            artist = credit.get("artist", {})
            name = credit.get("name") or artist.get("name", "?")
            mbid = artist.get("id", "no-mbid")
            join = credit.get("joinphrase", "")
            parts.append(f"{name} [{mbid}]{join}")
        else:
            parts.append(str(credit))
    pad = "  " * indent
    print(f"{pad}{'Artist Credit':<30} {''.join(parts)}")


def inspect_release_group(rgid: str) -> None:
    print("\n" + "=" * 70)
    print(f"RELEASE GROUP: {rgid}")
    print("=" * 70)

    print("\n[1/3] Fetching release group metadata...")
    rg = mb_get(
        f"release-group/{rgid}", {"inc": "artists+releases+tags+ratings", "fmt": "json"}
    )

    print()
    fmt("Title:", rg.get("title"))
    fmt("MBID:", rg.get("id"))
    fmt("Primary Type:", rg.get("primary-type", "not set"))
    fmt("Secondary Types:", ", ".join(rg.get("secondary-types", [])) or "none")
    fmt("First Release Date:", rg.get("first-release-date", "unknown"))
    fmt("Disambiguation:", rg.get("disambiguation", "none"))

    print()
    print("  --- Artist Credits ---")
    print_artist_credits(rg.get("artist-credit", []), indent=1)

    tags = rg.get("tags", [])
    if tags:
        tag_list = ", ".join(
            f"{t['name']} ({t['count']})"
            for t in sorted(tags, key=lambda x: -x["count"])[:10]
        )
        fmt("Tags:", tag_list, indent=1)

    rating = rg.get("rating", {})
    if rating:
        fmt(
            "Rating:",
            f"{rating.get('value', '?')}/100 ({rating.get('votes-count', 0)} votes)",
            indent=1,
        )

    releases = rg.get("releases", [])
    print(f"\n[2/3] Found {len(releases)} release(s) in this group.")

    print("\n" + "-" * 70)
    print("RELEASES SUMMARY")
    print("-" * 70)
    for i, rel in enumerate(releases, 1):
        print(f"\n  Release #{i}")
        fmt("Title:", rel.get("title"), indent=2)
        fmt("MBID:", rel.get("id"), indent=2)
        fmt("Date:", rel.get("date", "unknown"), indent=2)
        fmt("Country:", rel.get("country", "unknown"), indent=2)
        fmt("Status:", rel.get("status", "not set"), indent=2)
        fmt("Disambiguation:", rel.get("disambiguation", "none"), indent=2)

    # Fetch full detail on first release as a deep example
    if releases:
        print(
            f"\n[3/3] Fetching full detail for first release ({releases[0]['id']})..."
        )
        rel_detail = mb_get(
            f"release/{releases[0]['id']}",
            {
                "inc": "artists+labels+recordings+release-groups+media+artist-credits+isrcs",
                "fmt": "json",
            },
        )

        print("\n" + "-" * 70)
        print("FIRST RELEASE — FULL DETAIL")
        print("-" * 70)
        fmt("Title:", rel_detail.get("title"))
        fmt("MBID:", rel_detail.get("id"))
        fmt("Status:", rel_detail.get("status", "not set"))
        fmt("Date:", rel_detail.get("date", "unknown"))
        fmt("Country:", rel_detail.get("country", "unknown"))
        fmt("Barcode:", rel_detail.get("barcode", "none"))
        fmt("Packaging:", rel_detail.get("packaging", "unknown"))
        fmt(
            "Language:",
            rel_detail.get("text-representation", {}).get("language", "unknown"),
        )

        print()
        print("  --- Artist Credits ---")
        print_artist_credits(rel_detail.get("artist-credit", []), indent=1)

        labels = rel_detail.get("label-info", [])
        if labels:
            print("\n  --- Label Info ---")
            for li in labels:
                label = li.get("label") or {}
                fmt("Label:", label.get("name", "unknown"), indent=2)
                fmt("Catalog #:", li.get("catalog-number", "none"), indent=2)

        media = rel_detail.get("media", [])
        print(f"\n  --- Media ({len(media)} disc(s)) ---")
        for disc in media:
            disc_num = disc.get("position", "?")
            disc_fmt = disc.get("format", "unknown")
            tracks = disc.get("tracks", [])
            print(f"\n    Disc {disc_num} [{disc_fmt}] — {len(tracks)} tracks")
            for t in tracks:
                rec = t.get("recording", {})
                length_ms = t.get("length") or rec.get("length") or 0
                length_str = (
                    f"{length_ms // 60000}:{(length_ms % 60000) // 1000:02d}"
                    if length_ms
                    else "?:??"
                )
                artist_credit = t.get("artist-credit", rec.get("artist-credit", []))
                track_artist = (
                    "".join(
                        (c.get("name") or c.get("artist", {}).get("name", ""))
                        + c.get("joinphrase", "")
                        if isinstance(c, dict)
                        else str(c)
                        for c in artist_credit
                    )
                    or "same as album"
                )
                print(
                    f"      {t.get('number', '?'):>4}. {t.get('title', rec.get('title', '?')):<50} {length_str}  [{track_artist}]"
                )

    print("\n" + "=" * 70)
    print("LIDARR MATCHING NOTES")
    print("=" * 70)
    primary_type = rg.get("primary-type", "")
    secondary_types = rg.get("secondary-types", [])
    status = releases[0].get("status", "") if releases else ""

    print(f"\n  Primary type  : {primary_type or 'NOT SET ⚠'}")
    print(f"  Secondary types: {', '.join(secondary_types) or 'none'}")
    print(f"  First release status: {status or 'NOT SET ⚠'}")

    if not primary_type:
        print(
            "\n  ⚠  WARNING: No primary type set. Lidarr metadata profiles filter by type."
        )
        print(
            "     This release group will be INVISIBLE to Lidarr unless your profile includes blank types."
        )
    if status.lower() not in ("official", ""):
        print(
            f"\n  ⚠  WARNING: Status is '{status}'. Lidarr defaults to Official releases only."
        )

    credits = rg.get("artist-credit", [])
    artist_names = [
        (c.get("name") or c.get("artist", {}).get("name", ""))
        for c in credits
        if isinstance(c, dict) and "artist" in c
    ]
    if len(artist_names) > 1 or any(
        "&" in n or " and " in n.lower() for n in artist_names
    ):
        print(
            "\n  ⚠  COLLABORATION DETECTED: Artist credit has multiple entities or uses '&'."
        )
        print(
            "     Lidarr tracks albums under one artist. Check which MBID Lidarr uses for Frank Zappa:"
        )
        for c in credits:
            if isinstance(c, dict) and "artist" in c:
                a = c["artist"]
                print(f"       Name: {a.get('name')}  MBID: {a.get('id')}")
        print(
            "     In Lidarr, go to Frank Zappa's artist page and compare his MBID to the above."
        )

    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mb_debug.py <release-group-mbid>")
        print("Example: python mb_debug.py 0a59decb-2c3f-3ded-bfda-f7b4a2b09bfd")
        print("\nTo find a release group MBID:")
        print(
            "  beet list -a -f '$album | $mb_releasegroupid' albumartist:'Frank Zappa'"
        )
        sys.exit(1)

    rgid = sys.argv[1].strip()
    inspect_release_group(rgid)
