from __future__ import annotations

from pathlib import Path

from eib_spellchecker.inference.service import ArtifactSpellChecker


def launch_demo(artifact_dir: str | Path) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError('La demo requiere Gradio. Instala el extra con: pip install -e .[demo]') from exc

    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)

    def _correct(text: str):
        corrected, details = checker.correct_text(text)
        changed = [
            {"original": d.original, "corrected": d.corrected, "changed": d.changed}
            for d in details if d.changed
        ]
        return corrected, changed

    with gr.Blocks(title='eib-spellchecker demo') as demo:
        gr.Markdown(
            f"# eib-spellchecker demo\n\n**Idioma:** {checker.language}  \n**Backend:** {checker.backend.__class__.__name__}"
        )
        source = gr.Textbox(label='Texto de entrada', lines=5)
        target = gr.Textbox(label='Texto corregido', lines=5)
        changes = gr.JSON(label='Cambios detectados')
        run = gr.Button('Corregir')
        run.click(_correct, inputs=[source], outputs=[target, changes])
    demo.launch()
