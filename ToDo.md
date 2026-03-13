Als Nächstes kommt klar `Wave A: chars2vec alleine`.

Warum:
- `LSPO` ist wieder grün und sichtbar.
- der Code ist jetzt bereinigt, also kein schlechter Ballast mehr im Pfad.
- `chars2vec` ist noch der größte unerwartete Rückschritt.
- `Clustering` fassen wir absichtlich erst danach an, damit wir keine Änderungen stacken.

Die Reihenfolge ist jetzt:
1. `chars2vec` gezielt optimieren, ohne Clustering mitzunehmen.
2. `LSPO` laufen lassen mit Tag `perf_pkg2_chars_v1`.
3. `ADS Full` laufen lassen als `bench_full_perf_pkg2_chars_v1`.
4. vergleichen:
   - wenn `LSPO` grün und `ADS` schneller als [bench_full_v22_fix2](/home/ubuntu/Author_Name_Disambiguation/artifacts/exports/bench_full_v22_fix2): promoten
   - wenn nicht: Code-Rollback, ADS-Run mit [prune_infer_run.py](/home/ubuntu/Author_Name_Disambiguation/scripts/ops/prune_infer_run.py) auf JSON-only schrumpfen
5. erst dann `Wave B: Clustering alleine`

Was wir ausdrücklich noch nicht machen:
- kein `SPECTER`
- kein `Pair-Scoring`
- kein gemischtes `chars2vec + clustering`-Paket

Kurz: der nächste sinnvolle Schritt ist jetzt die **isolierte chars2vec-Runde**. Wenn du willst, setze ich die jetzt direkt um.