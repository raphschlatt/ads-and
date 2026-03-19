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