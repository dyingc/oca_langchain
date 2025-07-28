import streamlit as st
import streamlit.components.v1 as components
import base64
import textwrap
from markdown_it import MarkdownIt
from markdown_it.renderer import RendererProtocol
from markdown_it.common.utils import escapeHtml

def render_copy_button(text_to_copy: str, component_key: str):
    """
    Renders a copy button for a given text.
    """
    encoded_text = base64.b64encode(text_to_copy.encode("utf-8")).decode("utf-8")
    component_html = f"""
    <div style="display: flex; justify-content: flex-end; margin-top: -45px; margin-right: 5px;">
        <style>
            .copy-btn-whole {{
                border: none;
                background: #373737;
                color: #ccc;
                cursor: pointer;
                padding: 6px 8px;
                border-radius: 4px;
                font-size: 12px;
                transition: background 0.2s, color 0.2s;
            }}
            .copy-btn-whole:hover {{ background: #4a4a4a; color: white; }}
        </style>
        <button class="copy-btn-whole" onclick="copyToClipboard(this, '{encoded_text}')">
            <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTUgMjFWNWgzdjE0SDVabTQtNEgxN1YzaC04djE0Wm0tNCA1VjBoMTB2M2gtN3YxNmgyVjZoN3YxNkgzWiIvPjwvc3ZnPg==" alt="Copy">
        </button>
        <script>
            if (!window.copyToClipboard) {{
                window.copyToClipboard = function(element, b64text) {{
                    const decodedText = atob(b64text);
                    navigator.clipboard.writeText(decodedText).then(function() {{
                        const originalText = element.innerHTML;
                        element.innerText = 'Copied!';
                        setTimeout(() => {{ element.innerHTML = originalText; }}, 2000);
                    }}, function(err) {{
                        console.error('Could not copy text: ', err);
                        alert("Copy failed. Please copy manually.");
                    }});
                }}
            }}
        </script>
    </div>
    """
    components.html(component_html, height=35)

def add_copy_to_code_blocks(markdown_text: str, key_prefix: str) -> str:
    """
    Injects a copy button into each code block of a markdown string.
    """
    md = MarkdownIt().enable('html_block').enable('html_inline')

    def custom_fence_renderer(self: RendererProtocol, tokens, idx, options, env):
        token = tokens[idx]
        info = token.info.strip() if token.info else ""
        lang_name = info.split(maxsplit=1)[0] if info else ""

        if options.highlight:
            highlighted_code = options.highlight(token.content, lang_name, "")
        else:
            highlighted_code = escapeHtml(token.content)

        if not highlighted_code.startswith('<pre'):
            lang_class = f' class="language-{lang_name}"' if lang_name else ''
            highlighted_code = f'<pre><code{lang_class}>{highlighted_code}</code></pre>'

        encoded_content = base64.b64encode(token.content.encode("utf-8")).decode("utf-8")
        button_html = f'<button class="copy-btn-code" onclick="copyToClipboard(this, \'{encoded_content}\')"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTUgMjFWNWgzdjE0SDVWmTQtNEgxN1YzaC04djE0Wm0tNCA1VjBoMTB2M2gtN3YxNmgyVjZoN3YxNkgzWiIvPjwvc3ZnPg==" alt="Copy"></button>'

        return f'<div class="code-container">{button_html}{highlighted_code}</div>'

    md.add_render_rule("fence", custom_fence_renderer)

    # Use textwrap.dedent to remove leading whitespace from the multiline string
    shared_html = textwrap.dedent("""
    <style>
        .code-container { position: relative; margin-top: 1em; margin-bottom: 1em; }
        .copy-btn-code {
            position: absolute; top: 0.5em; right: 0.5em; z-index: 1;
            border: none; background: #373737; color: #ccc; cursor: pointer;
            padding: 6px 8px; border-radius: 4px; font-size: 12px;
            opacity: 0.5; transition: opacity 0.2s, background 0.2s, color 0.2s;
        }
        .code-container:hover .copy-btn-code { opacity: 1; }
        .copy-btn-code:hover { background: #4a4a4a; color: white; }
    </style>
    <script>
        if (!window.copyToClipboard) {
            window.copyToClipboard = function(element, b64text) {
                const decodedText = atob(b64text);
                navigator.clipboard.writeText(decodedText).then(function() {
                    const originalText = element.innerHTML;
                element.innerText = 'Copied!';
                setTimeout(() => { element.innerHTML = originalText; }, 2000);
                }, function(err) {
                    console.error('Could not copy text: ', err);
                });
            }
        }
    </script>
    """)

    rendered_markdown = md.render(markdown_text)
    return shared_html + rendered_markdown
