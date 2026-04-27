# FAIB NEXUS

FAIB NEXUS ist ein experimentelles Python-Framework zur Analyse von Zeitreihen über Fraktalebenen, Zustandswechsel, MACD-Struktur, Nexus-Matrix sowie optionale Visualisierungen und Forschungs-Module.

Der aktuelle Stand ist bewusst praxisnah und iterativ gewachsen: Daten werden über konfigurierbare Loader geladen, in `urdaten` überführt, zu mehreren Fraktalebenen verarbeitet und anschließend durch optionale Analyse- und Visualisierungs-Module ausgewertet.

## Ziel des Projekts

Das Projekt untersucht, wie sich eine Rohzeitreihe in diskrete Fraktalstufen, Zustandsdynamiken und strukturierte Long/Short-Dominanz übersetzen lässt.

Darauf aufbauend entstehen Module für Nexus-Matrix, Fusion, Verlagerung, Interferenz, DFA/Hurst/MFDFA sowie mehrere Debug- und Forschungsansichten.

## Projektlogik

Der Ablauf im aktuellen Stand ist im Kern wie folgt aufgebaut:

1. In `config.py` werden Datenquellen, Fraktal-Level, MACD-Parameter, Ausgabeordner und Speicherformat festgelegt.
2. `loaders.py` lädt die jeweilige Zeitreihe, etwa Solar-, NASDAQ- oder ERA5-Daten, abhängig vom konfigurierten Loader.
3. `urdaten.py` erzeugt daraus die normalisierte Basisreihe mit `cumheight` als zentraler Referenzbahn für die weitere Fraktalverarbeitung.
4. `fraktale.py` baut daraus alle konfigurierten Fraktalebenen auf, inklusive `closeF`, `dot`, `nblocks` und optionaler Folgeinformationen.
5. `macddom.py` berechnet klassischen MACD, eventbasierten MACD auf Fraktalwechseln sowie binäre Dominanzzustände.
6. `nexus.py` verdichtet die Dominanz je Fraktal zu Matrix, Fusion und Verlagerung.
7. `main.py` steuert den Gesamtprozess, speichert Basisergebnisse und aktiviert je nach Flags zusätzliche Analyse- oder Visualisierungs-Module.

## Wichtige Dateien

| Datei | Rolle |
|---|---|
| `config.py` | Zentrale Konfiguration für Dateien, Systeme, Fraktal-Level, MACD und Output. |
| `main.py` | Hauptsteuerung des gesamten Workflows und Aktivierung optionaler Module. |
| `loaders.py` | Lädt verschiedene Datenquellen über benannte Loader-Funktionen. |
| `urdaten.py` | Baut aus der Rohserie die Basisdaten für `cumheight` und Folgeanalysen. |
| `fraktale.py` | Erzeugt Fraktalebenen aus `cumheight`. |
| `macddom.py` | Berechnet MACD, eventbasierten MACD und Dominanzzustände. |
| `nexus.py` | Erstellt Nexus-Matrix, Fusion und Verlagerung aus den Fraktalzuständen. |
| `mainstorage.py` | Speichert und lädt Ergebnisse als CSV oder Parquet je nach Konfiguration. |
| `viz*.py` | Sammlung experimenteller Visualisierungen für Debugging, Forschung und Interpretation. |

## Datenquellen

Im aktuellen Stand sind in `config.py` mehrere Datenpfade vorgesehen, darunter Solar-Daten, NASDAQ-Einzeldaten, NASDAQ-Multiday-Ordner und ERA5-NetCDF-Daten.

Die aktiven Systeme werden über die Liste `SYSTEMS` definiert, wobei jeder Eintrag einen Namen, eine Datei bzw. einen Ordner und einen Loader-Namen enthält.

## Steuerung über Flags

`main.py` enthält eine Reihe von Schaltern, mit denen Basisplots, Zusatzanalysen und experimentelle Visualisierungen gezielt aktiviert oder deaktiviert werden können.
Dazu gehören unter anderem Flags für RS, DFA, MFDFA, Hurst, Compare, Panels, Interaktiv, Plotly, Verlagerung, Matrix, Interferenz und Hurst-Visualisierung.

## Speicherlogik

Die Basisdaten können je nach `STORAGE_FORMAT` als CSV oder Parquet gespeichert werden, wobei `mainstorage.py` das Speichern und spätere Wiederladen zentral übernimmt.

Der Modus in `main.py` erlaubt dabei den Wechsel zwischen kompletter Neuberechnung und dem Arbeiten auf bereits gespeicherten Ergebnissen.

## Charakter des Repos

Dieses Repository ist kein klassisches Endprodukt, sondern eher ein forschungsnahes Arbeitsframework mit experimentellen Modulen, wachsender Struktur und mehreren parallel entwickelten Auswertungswegen.

Gerade deshalb eignet es sich gut für eine schrittweise Verbesserung mit Cursor AI: Module können einzeln bereinigt, vereinheitlicht und dokumentiert werden, ohne die gesamte Logik auf einmal umzubauen.
## Sinnvolle nächste Verbesserungen

- Imports und Modulnamen vereinheitlichen, insbesondere bei optionalen Visualisierungen.
- Gemeinsame Hilfsfunktionen wie Fraktalfilterung und Sampling zentralisieren.
- Spaltennamen konsequent vereinheitlichen, etwa `cumheight` gegenüber `cum_height`.
- README, Modul-Dokus und Beispiel-Workflows ausbauen.
- Optional eine klarere Trennung zwischen Kernlogik, Forschungsmodulen und Visualisierung schaffen.

## Nutzungsidee mit Cursor AI

Am besten wird dieses Projekt nicht in einem einzigen großen Refactor verändert, sondern in kleinen, kontrollierten Schritten.[1]

Ein sinnvoller Workflow ist: erst ein Modul analysieren, dann minimale Patches definieren, anschließend testen und erst danach das nächste Modul bereinigen.[1]
