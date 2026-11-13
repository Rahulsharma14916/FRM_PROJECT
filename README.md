MA634 Assignment 
Submitted by: Rahul (2023EEB1238), Lakshya Vajpayee (2023CEB1037), Sargun Singh (2023CEB1046)
Submitted to: Prof. Puneet Pasricha, Department of Mathematics, IIT Ropar

Email deliverables (subject: "MA634 Assignment 2023EEB1238")
  Codes.py       code (rolling backtest, no look-ahead)
  Assignment.pdf  report (compiled from the Overleaf project)
  outputs/returns_matrix_nx5.csv        n x 5 matrix (54 x 5): GMV, MV, EW, Active, NIFTY50

Supporting files
  outputs/weights_GMV|MV|EW|Active.csv  weights at the end of each formation window
  outputs/window_details.csv, performance_table.csv, cumulative_returns.png, diagnostics.png
  make_latex_tables.py                  converts outputs/*.csv into LaTeX tables
  overleaf/                             Overleaf project: upload as a zip, main file = main.tex

Reproduce
  pip install numpy pandas scipy matplotlib
  python Assignment_2023EEB1238_Codes.py <folder with Stocks_data.csv and market_Factor_risk_Free.csv>
  python make_latex_tables.py        (then compile overleaf/main.tex)
