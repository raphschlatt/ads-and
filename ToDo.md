`wave_b_v1` ist als fehlgeschlagener Clustering-Kandidat abgeschlossen.

Gesicherter Stand:
- `LSPO` war nur die Vorqualifikation und blieb innerhalb der Toleranz.
- `ADS Full` war die operative Entscheidungsebene und ist klar regressiv ausgefallen.
- Fuer `bench_full_wave_b_v1` existiert jetzt eine formale `keep_baseline`-Decision.
- Der Run wurde auf die dokumentierte JSON-only-Retention reduziert.
- Die aktive infer-Baseline bleibt `bench_full_v22_fix2`.

Was daraus folgt:
1. `wave_b_v1` wird nicht promotet.
2. Diese Konfiguration ist nicht mehr der laufende Arbeitszweig.
3. Der naechste Performance-Schritt wird separat und von der sauberen Baseline aus neu geplant.

Was wir jetzt bewusst nicht tun:
- kein Weiteriterieren auf dem entfernten `wave_b_v1`-Configpfad
- kein gemischtes Paket aus neuen Clustering- und chars2vec-Aenderungen
- kein neuer Full-Run, bevor der naechste Versuch inhaltlich neu begruendet ist

Kurz: erst Close-out und Baseline-Hygiene, dann neue Planung aus sauberem Zustand.

`perf_pkg2_chars_v1` ist als chars2vec-Kandidat jetzt ebenfalls eingeordnet.

Gesicherter Stand:
- Der chars2vec-Microbenchmark war stark positiv: `predict+auto` war auf dem ADS-Sample grob `7.6x` schneller als `predict+32`.
- Der `LSPO Gate Run` ist trotzdem nicht durch das aktuelle No-Drift-Gate gekommen.
- Ein `ADS Full Candidate Run` fuer `perf_pkg2_chars_v1` wurde deshalb bewusst nicht gestartet.
- Die Details und Messwerte stehen in `docs/experiments/perf_pkg2_chars_v1.md`.

Was daraus folgt:
1. `perf_pkg2_chars_v1` wird in der jetzigen Form nicht promotet.
2. Grosze chars2vec-Batches sind fuer dieses Repo nicht automatisch ein reiner Runtime-Gewinn, sondern ein qualitaetsrelevanter Eingriff.
3. Vor dem naechsten promoteten Inferenz-Experiment sollte der historische chars2vec-Defaultpfad wieder der Referenzzustand sein.

Was wir als Naechstes sinnvoll tun:
1. chars2vec auf den historischen Referenzpfad zurueckstellen
2. den naechsten chars2vec-Versuch nur noch als Exact-Path-Optimierung planen
3. erst dann einen neuen `LSPO Gate Run` ansetzen
