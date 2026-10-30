# ChessCam – DAT158

Marius Phillips, høsten 2026.

ChessCam bruker kamera og maskinlæring til å lese et fysisk sjakkbrett og registrere trekk. Faglærer har godkjent at treningskode og en innspilt demo leveres uten hele appens kildekode.

- **Rapport:** [PDF](REPORT.pdf), [Word](REPORT.docx) eller [Markdown](REPORT.md).
- **Treningskode:** [train.py](train.py), et forenklet utdrag av prosjektets LeYOLO-trening.
- **Resultater:** [results.json](results.json), hentet fra den lagrede treningsloggen.
- **Prøv appen:** [camerachess.com](https://camerachess.com/).
- **Video:** Sett inn lenken til demoen her.

## Kjøring

Bruk Python 3.12 og et eget miljø:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --data data --weights startmodell.pt --device 0
```

`--device 0` bruker CUDA-GPU. Bruk `--device cpu --batch 8` på en maskin uten GPU. `--epochs 1` gir én treningsrunde, og `--dry-run` viser innstillingene uten å starte trening.

Koden forventer ferdig klargjorte YOLO-datasett under `data/upstream/`, `data/custom/`, `data/external-v1/` og `data/real-v2/`. Hvert datasett skal ha `images/train/`, `images/val/`, `labels/train/` og `labels/val/`.

Hver merkefil har samme grunnnavn som bildet. Én linje per brikke har formatet `klasse midt_x midt_y bredde høyde`, med koordinater mellom 0 og 1. Klasserekkefølgen står i `CLASSES` i koden. Data må være renset, klassene samordnet og opptak fordelt mellom trening og validering før kjøring.

Startvektene må være et kompatibelt LeYOLO-checkpoint. Den rapporterte `real-v2`-runden startet fra prosjektets tidligere `external-v1`-modell. Bilder og vekter følger ikke med, så eksakt gjenskaping av denne runden krever tilgang til dem.

Utdraget beholder modellinnstillingene, vektingen av datakilder, treningen og separat validering. Skylagring, nedlasting og konvertering av rådata er utelatt. Den nye kjøringen lagrer beste modell og `validation.json` under `runs/`. Kilder og forklaringer finnes i rapporten. Koden beholder prosjektets [AGPL-3.0-only-lisens](LICENSE.md).

# chesscam_dat158_submission
