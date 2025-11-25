#!/usr/bin/env python3
"""
Dynamic DZI tile server for WSI files.

Usage:
    uv run python scripts/webview.py --wsi-file slide.ndpi
    uv run python scripts/webview.py -w slide.ndpi --port 8080
"""

import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import Field
from pydantic_autocli import AutoCLI, param

from wsi_toolbox.wsi_files import PyramidalWSIFile, create_wsi_file


class WebViewArgs(AutoCLI.CommonArgs):
    wsi_file: Path = param(..., s="-w", description="Path to WSI file (.ndpi, .svs, etc.)")
    port: int = param(8000, s="-p", description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")
    tile_size: int = Field(default=256, description="DZI tile size")
    overlap: int = Field(default=0, description="DZI tile overlap")
    jpeg_quality: int = Field(default=90, description="JPEG quality (0-100)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WSI Viewer - {name}</title>
  <script src="https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1a1a1a;
      color: #fff;
    }}
    header {{
      padding: 12px 20px;
      background: #2a2a2a;
      border-bottom: 1px solid #3a3a3a;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    header h1 {{
      font-size: 16px;
      font-weight: 500;
    }}
    header .info {{
      font-size: 13px;
      color: #888;
    }}
    #viewer {{
      width: 100vw;
      height: calc(100vh - 48px);
      background: #000;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{name}</h1>
    <span class="info">{width} x {height} px</span>
  </header>
  <div id="viewer"></div>
  <script>
    OpenSeadragon({{
      id: "viewer",
      prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
      tileSources: "/{dzi_name}.dzi",
      showNavigationControl: true,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      animationTime: 0.5,
      blendTime: 0.1,
      constrainDuringPan: true,
      maxZoomPixelRatio: 2,
      minZoomLevel: 0.5,
      visibilityRatio: 1,
      zoomPerScroll: 2,
      timeout: 120000
    }});
  </script>
</body>
</html>
"""


class CLI(AutoCLI):
    def run_webview(self, args: WebViewArgs):
        """Start dynamic DZI tile server for WSI viewing"""
        wsi_path = args.wsi_file
        if not wsi_path.exists():
            print(f"Error: File not found: {wsi_path}")
            return False

        print(f"Loading WSI: {wsi_path}")
        wsi = create_wsi_file(str(wsi_path))

        if not isinstance(wsi, PyramidalWSIFile):
            print(f"Error: {wsi_path} is not a pyramidal WSI file (DZI not supported)")
            return False

        width, height = wsi.get_original_size()
        max_level = wsi.get_dzi_max_level()
        dzi_name = wsi_path.stem

        print(f"Size: {width}x{height}, Max level: {max_level}")

        app = FastAPI(title="WSI DZI Server")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTML_TEMPLATE.format(
                name=wsi_path.name,
                width=width,
                height=height,
                dzi_name=dzi_name,
            )

        @app.get(f"/{dzi_name}.dzi")
        def get_dzi():
            xml = wsi.get_dzi_xml(args.tile_size, args.overlap, "jpeg")
            return Response(content=xml, media_type="application/xml")

        @app.get(f"/{dzi_name}_files/{{level}}/{{col}}_{{row}}.jpeg")
        def get_tile(level: int, col: int, row: int):
            try:
                tile = wsi.get_dzi_tile(level, col, row, args.tile_size, args.overlap)
                img = Image.fromarray(tile)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=args.jpeg_quality)
                buf.seek(0)
                return Response(content=buf.getvalue(), media_type="image/jpeg")
            except Exception as e:
                print(f"Error getting tile {level}/{col}_{row}: {e}")
                return Response(status_code=404)

        print(f"Starting server at http://{args.host}:{args.port}/")
        print(f"DZI endpoint: http://localhost:{args.port}/{dzi_name}.dzi")
        uvicorn.run(app, host=args.host, port=args.port)


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
