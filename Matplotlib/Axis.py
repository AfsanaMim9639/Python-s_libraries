{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cb4333bb-d288-4cd8-8beb-ab4eb5fbbf75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhYAAAGdCAYAAABO2DpVAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA7jElEQVR4nO3deVzUBf7H8fcMA8iNeAGK94ECmppXaaVphqZmWal0mGaXdllmbltrtZuatZXmumWlZZKVmVl5lFqpmfcFnuB9gHhxywAz398fFpu/skS/MDPwej4e/AHfYeYzjjAvvqfFMAxDAAAAJrC6egAAAFBxEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATGMr7wd0Op06duyYgoKCZLFYyvvhAQDAJTAMQzk5OYqMjJTVeuH1EuUeFseOHVNUVFR5PywAADDB4cOHVadOnQsuL/ewCAoKknRusODg4PJ+eAAAcAmys7MVFRVV8j5+IeUeFr9u/ggODiYsAADwMH+1GwM7bwIAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADANYQEAAExDWAAAANMQFgAAwDSEBQAAMA1hAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTlCosxo0bJ4vFct5HdHR0Wc0GAAA8jK203xATE6OlS5f+7w5spb4LAABQQZW6Cmw2m8LDw8tiFgAA4OFKvY9FSkqKIiMj1bBhQyUkJOjQoUN/enu73a7s7OzzPgAAgPn+/d0evbk0RU6n4bIZShUWHTp00MyZM7V48WJNmzZN+/fvV5cuXZSTk3PB7xk/frxCQkJKPqKioi57aAAAcL7vd2do8rIUvb50j37ed8plc1gMw7jkrMnMzFS9evX073//W8OGDfvD29jtdtnt9pLPs7OzFRUVpaysLAUHB1/qQwMAgF8czTyr3pNXKjO/SHd1rKeXbo41/TGys7MVEhLyl+/fl7XnZWhoqJo2barU1NQL3sbX11e+vr6X8zAAAOACCoudGjF7kzLzi9SyToj+flNzl85zWeexyM3N1d69exUREWHWPAAAoBReXrhTWw5nKriKTVMHt5Gvzcul85QqLJ566in9+OOPOnDggFavXq3+/fvLy8tLgwYNKqv5AADABXyzLU0zVx+QJP379isUFebv2oFUyk0hR44c0aBBg3Tq1CnVqFFDnTt31po1a1SjRo2ymg8AAPyBfSdyNebzbZKkB69tpO4tarl4onNKFRZz5swpqzkAAMBFKihy6OHZm5RrL1b7BmF66oamrh6pBNcKAQDAwzz/ZbJ2peeoeqCP3hrUWjYv93k7d59JAADAX/p0w2F9uuGIrBZp8sDWqhlcxdUjnYewAADAQ+xMy9Zz85MlSaN6NNVVjau7eKLfIywAAPAAOQVFenj2JtmLnbquWQ09fF1jV4/0hwgLAADcnGEYeubzJO0/mafIkCp6/fYrZLVaXD3WHyIsAABwcx+sPqBvktLk7WXRWwltVDXAx9UjXRBhAQCAG9t86Iz+tXCnJGlsfHO1qVvVxRP9OcICAAA3dSavUCNmb1KRw1CvuHDde3V9V4/0lwgLAADckNNp6IlPt+hYVoEaVA/QxFtbymJxz/0qfouwAADADU37ca9+2H1Cvjar/pPQRkFVvF090kUhLAAAcDOr957Ua9/uliS91C9WzSOCXTzRxSMsAABwIxnZBXr04y1yGtKAtnV0e7soV49UKoQFAABuotjh1MiPN+tkrl3R4UF6qV+sq0cqNcICAAA38dp3e7Ru/2kF+tr0n4Q28vPxcvVIpUZYAADgBpbtPK5pP+yVJE28taUa1gh08USXhrAAAMDFDp/O16hPt0qShlxVX71bRrh4oktHWAAA4EL2YodGJG5S1tkitYoK1d96NXf1SJeFsAAAwIX+9c1ObTuSpVB/b00d3Fo+Ns9+a/bs6QEA8GBfbT2mD38+KEl6/Y4rVKeqv4snunyEBQAALrD3RK6e+XybJGlE10bq2qymiycyB2EBAEA5yy8s1kMfbVReoUMdG4bpie5NXT2SaQgLAADKkWEY+vv8ZO05nqsaQb6aPKi1bF4V5+244jwTAAA8wCfrD2vepqOyWqQpg1qrZlAVV49kKsICAIBysv1Ylp5fsF2S9FTPZurYsJqLJzIfYQEAQDnILijSw7M3qbDYqeuja+rBaxq5eqQyQVgAAFDGDMPQ059t08FT+aod6qfXbm8lq9Xi6rHKxGWFxYQJE2SxWPT444+bNA4AABXPe6v2a/H2dHl7WTQ1oY1C/X1cPVKZueSwWL9+vd5++221bNnSzHkAAKhQNh48rQmLdkmSnrupha6ICnXtQGXsksIiNzdXCQkJmj59uqpWrWr2TAAAVAin8wo1MnGzip2G+rSK1F0d67l6pDJ3SWExYsQI9e7dW927d//L29rtdmVnZ5/3AaDiOXgqT5OW7NKudH7GAUlyOg09/skWpWUVqGGNAI2/JU4WS8Xcr+K3bKX9hjlz5mjTpk1av379Rd1+/PjxeuGFF0o9GADP4HAa+mD1AU1asltnixz68OeDmjWsQ4Vf3Qv8lbe+T9WKPSdUxduqaQltFehb6rdcj1SqNRaHDx/WY489ptmzZ6tKlYs7ocfYsWOVlZVV8nH48OFLGhSA+0nNyNVt/12tF7/eobNFDgVXsSmnoFh3vbtWGw+ecfV4gMusSjmp15fukST98+Y4NQsPcvFE5cdiGIZxsTeeP3+++vfvLy8vr5KvORwOWSwWWa1W2e3285b9kezsbIWEhCgrK0vBwcGXPjkAlyl2OPXOyn16Y2mKCoudCvS16W+9mqvvFZEaNnO91u4/rQAfL80c2l7t6oe5elygXKVnFaj35JU6lVeoge2iNOHWinGQw8W+f5cqLHJycnTw4MHzvnbvvfcqOjpaY8aMUWxsrGmDAXBPO9Oy9fTcbUo6miVJuq5ZDb3cP06RoX6Szl1c6b4PNmj13lPy9/HS+0PaVcizCwJ/pMjh1ODpa7T+wBm1iAjWvIevUhXvP/+D21Nc7Pt3qTb4BAUF/S4eAgICVK1atYuKCgCeq7DYqanfp2rq96kqdhoK8fPW8ze10C1tap+3Q5q/j03v3dNO98/aoJUpJzVkxjq9f087XdW4ugunB8rHq0t2a/2BMwrytek/CW0qTFSUBmfeBPCXth3JVJ8pq/TmshQVOw31jKml70Zdo1vb1vnDvdz9fLw0/e4rdW3TGioocuremeu1MuWECyYHys+329P19op9kqRJt7VU/eoBLp7INUq1KcQMbAoBPEdBkUOvL92j6Sv2yWlI1QJ89EK/GPWOi7iow+bsxQ499NEmLd+VIR+bVe/c1VbXNatZDpMD5evQqXz1nrJSOQXFGnp1Az3fp4WrRzLdxb5/s8YCwB/acOC0er25Um//eC4q+l0Rqe9GXaubWkZe9LH4vjYvTbuzjXq0qKXCYqfu/3Cjlu86XsaTA+WroMihhxM3KqegWG3qhuqZ+GhXj+RShAWA8+QXFmvcgu267e2fte9knmoG+Wr63VfqzYGtFRZQ+usb+Nq8NHVwG90YE65Ch1MPzNqo73YQF6g4Xvp6h5KPZquqv7feGtxGPrbK/dZauZ89gPP8lHpSPd9YoZmrD8gwpNuvrKPvRl2rHi1qXdb9+tismjK4tXrHRajIYeihjzZqcXK6SVMDrjN/81HNXntIFov0xsDWJUdHVWaV4zRgAP5UdkGRxi/cpY/XHZIk1Q710/hb4nRN0xqmPYa3l1VvDrxCXlaLFmw9phGJmzR5YGv1bhlh2mMA5SnleI7GzkuSJD3StbGuNfHnxZMRFkAlt3zXcf1tXrLSswskSXd1rKcx8dFlcvphm5dVr99xhWxWi+ZtPqpH52xWsdOpflfUNv2xgLKUZy/WQ7M36WyRQ1c3rqbHujd19Uhug7AAKqnM/EK9+NUOzdt8VJJUv5q/JtzassxPZuVltWjSba1ktVo0d+MRPfHJFjkNQ/1b1ynTxwXMYhiGnv0iSakZuaoV7Ks3B7aWl7XiX1zsYhEWQCW0ODlNf5+/XSdz7bJapGGdG2hUj2by8ymfk/l4WS165daWslktmrP+sEZ9ulUOpzSgLXEB95e47pDmbzkmL6tFbw1uo+qBvq4eya0QFkAlciLHrnELtuubpDRJUuOagXplQEu1qVu13GexWi16uX+cvKwWzV57SKPnbpXD6dQd7eqW+yzAxUo6kqUXFuyQJD3dsxnXwvkDhAVQCRiGoQVbj2ncgu06k18kL6tFD13bSI9c31i+NtedcthqteifN8fKZrXog58PasznSSp2GkroUM9lMwEXkpVfpIcTN6rQ4VT35rV0/zUNXT2SWyIsgAouPatAz36RpGW7MiRJzSOCNWlAS8XWDnHxZOdYLBaN6xsjL6tV7/+0X89+kSyH09Ddneq7ejSghGEYemruVh0+fVZ1qvrptdtaXfSJ4iobwgKooAzD0KcbDuufX+9Ujr1YPl5WPXp9Yz1wbSN5e7nXKWwsFoueu6m5bF4WvbNin57/cruKHYaGdm7g6tEASdL0lfv03Y7j8vGyalpCW4X4e7t6JLdFWAAV0OHT+Ro7L0mrUk9KklpFhWrSgJZqWivIxZNdmMVi0dj4aHlZLZr2w169+PUOOZyGhrO6GS62/sBpTVy8W5L0fJ8WiqvjHmv73BVhAVQgTqehj9Ye1IRFu5Rf6JCvzaqnbmimoZ0beMThcBaLRU/3bCab1aIpy1P1r4U7Vew09NB1jVw9Giqpk7l2jUzcJIfTUL8rIpXQgZ2L/wphAVQQ+0/maczcbVp34LQkqX39ME0c0FINPOzSzRaLRU/e0Ew2q1WvL92jiYt3yeF0amS3Jq4eDZWMw2nosTmbdTzbrsY1A/Vy/zj2q7gIhAXg4RxOQ++t2qfXvt0je7FT/j5eeiY+Wnd2qCerB6yluJDHujeRl1V69ds9evXbPSp2Gnrs+ib8Yke5eXNZin5KPSU/by9NS2ijgDI4G21FxL8S4MH2HM/R6LnbtPVwpiSpS5Pqerl/nKLC/F07mElGdmsiL6tVExfv0htLU+RwGhrVoylxgTK3Ys8JTVmeIkkaf0ucmrjx/knuhrAAPFCRw6n//rBXk5enqMhhKKiKTc/1bqHbrqxT4d50H7qukWxWi/61cKemLE9VsdPQ0z2bVbjnCfeRlnVWj3+yRYYhDe5QVze35lo2pUFYAB4m+WiWnp67TTvSsiVJ10fX1L/6xyk8pIqLJys7w69pKC+rRS9+vUPTftgrh9PQ2Pho4gKmK3I4NWL2Jp3OK1Rs7WA9f1MLV4/kcQgLwEPYix2asixV034898Za1d9b4/rGqG+ryErxBju0cwPZvCx6/svtemfFPhU5nHr+phaV4rmj/ExYtEubDmUqqIpN/xncVlW8XXdmWk9FWAAeYNOhM3p67jalZuRKknrHReiFfjGV7uJHd3eqLy+rRc9+kawZPx2Qw2nohb4xxAVMsTg5Te+t2i9JevW2VqpbrWLsq1TeCAvAjZ0tdOi1b3frvZ/2yzCk6oG+eqlfjOLjIlw9msskdKgnm9WiZ+Yl6cOfD8rhNPRSv1iPPgIGrnfgZJ5Gf7ZNknT/NQ3VMybcxRN5LsICcFNr9p3SmM+36eCpfEnSLW1q6/mbWijU38fFk7neHe3qystq1ei5WzV77SE5nIZe7h9HXOCSFBQ59PDsTcqxF6td/aoa3bOZq0fyaIQF4GZy7cWauGiXZq05KEmKCKmil/vHqWt0TRdP5l4GtK0jL6v05KdbNWf9YRU7DU28taVHnGEU7uWFr7ZrR1q2qgX4aMqgNm53LR1PQ1gAbmTFnhMaOy9JRzPPSpIGta+rsb2iFVyFCx79kf6t68jLatUTn2zR3I1H5HAaevW2VsQFLtrnG4/o43WHZbFIbw5sXaGPriovhAXgBrLyi/TPb3bos41HJElRYX6acEtLXd24uosnc399W0XKy2LRY3M264vNR+VwGvr37a1k469O/IXd6Tl6dn6SJOmx65uocxN+3sxAWAAu9t2O43r2iyRl5NhlsUj3dKqv0T2bcfrgUujdMkJeVotGJm7Sgq3H5HAaemPgFazSxgXl2ov10OyNKihyqkuT6nqEa9GYplQ/ddOmTVPLli0VHBys4OBgderUSYsWLSqr2YAK7XReoR79eLOGf7hBGTl2NaweoM8e6KRxfWOIiktwY2y4pt3ZVt5eFn2TlKZHEjersNjp6rHghgzD0Nh5Sdp3Ik/hwVX0xh1XsPnMRKUKizp16mjChAnauHGjNmzYoG7duqlfv37avn17Wc0HVDiGYejrbcfU498/asHWY7JapAevbaSFj3XRlfXDXD2eR+vRopbevqutfLysWrw9XQ/P3iR7scPVY8HNfLTmoL7aekw2q0VTE1qrWiU7H0xZsxiGYVzOHYSFhWnSpEkaNmzYRd0+OztbISEhysrKUnBw8OU8NOBxMrIL9NyXyVqy/bgkqVmtIL0yoKVaRYW6drAK5ofdGbp/1kYVFjvVLbqm/pPQhjMoQpK09XCmBvx3tYochv7eu7nu69LQ1SN5jIt9/77kDZAOh0Nz5sxRXl6eOnXqdMHb2e12ZWdnn/dRFp7/Mll/n5+k1XtPqtjB6k+4F8MwNHfjEXX/949asv24bFaLHru+ib56pDNRUQaua1ZT79/TTr42q5bvytADszaqoIg1F5VdZn6hHp69SUUOQz1jamlY5wauHqlCKvUai6SkJHXq1EkFBQUKDAxUYmKievXqdcHbjxs3Ti+88MLvvm7mGouzhQ61/ed3yi8894sjLMBHPWNqKT42Qp0aVWMHLrjU0cyz+tu8JP2454QkKbZ2sCYNaKXmEayxK2urU09q2AcbdLbIoS5Nquudu66Unw9rLiojp9PQ8A83aNmuDNWr5q8FIzsrxI/DuEvjYtdYlDosCgsLdejQIWVlZWnu3Ll699139eOPP6pFiz++Apzdbpfdbj9vsKioKFPDotjh1KrUk1qUlK4lO9KVmV9UsizEz1s3tKilXnERurpxdfnYiAyUD6fT0MfrD2n8wl3KtRfLx2bV492b6P4uDTkUshyt2XdKQ2euV36hQ1c1qqZ377lS/j7sHFvZTPthryYu3iUfm1VfPHyVYiJDXD2SxymzsPj/unfvrkaNGuntt982dbBLVeRwau2+01qYnKYlyek6lVdYsiyoik09mtdSfFyEujSpzjZXlJmDp/L0zOdJ+nnfKUlSm7qhemVAKzWuGejiySqn9QdOa8j765RX6FCHBmF6f0g7jrypRNbsO6WEd9fK4TQ0/pY4DWpf19UjeaRyC4tu3bqpbt26mjlzpqmDmcHhNLRu/2ktSk7TouR0ncj535qTAB8vXd+8lnrFhevapjVZPQpTOJyGPlh9QJOW7NbZIoeqeFv1dM9o3XNVfQ5nc7GNB89oyPvrSq4HMePe9gokLiq8jJwC9Z68Sidy7LqldW29dnsrroZ7icokLMaOHav4+HjVrVtXOTk5SkxM1MSJE7VkyRL16NHD1MHM5nQa2njojBYlpWtRcprSsgpKlvl5e6lbdE3Fx4Wra7Oa/CWDS5Kakaun527VpkOZkqRODatpwq1xqlctwLWDocSWw5m66721yikoVpu6oZo5tD2nS6/AHE5Dd767Vj/vO6WmtQI1f8TVbAa7DGUSFsOGDdOyZcuUlpamkJAQtWzZUmPGjLnoqCjNYGXJ6TS09UimFiWna2FSmo6cOVuyzNdm1bVNa6hXXIS6Na/JLx38pWKHU++s3Kc3lqaosNipQF+bxvaK1qB2dbnaphtKOpKlO99bq6yzRWoVFaoPh7ZnJ74K6tUlu/XW96ny9/HSgpGd2RR5mcptU0hpuUNY/JZhGEo+mq2FyWlamJRWcolqSfLxsqpLk+qKj4tQj+a1FOLPLx+cb2datp6eu01JR7MkSdc1q6GX+8cpMtTPxZPhz2w/lqU7312rM/lFiqsdolnD2nM5+grm+90ZunfGeknS5EGt1bdVpIsn8nyExSUwDEM703K0KDlN3ySlad+JvJJlNqtFVzeurl5x4erRIlxhAfwSqswKi52a+n2qpn6fqmKnoeAqNv2jT4xuaVOb7bceYmdathLeXavTeYVqERGs2fd1UFV+riuEo5ln1XvySmXmF+mujvX00s2xrh6pQiAsLpNhGErJyNXCpDQtSkrX7uM5Jcu8rBZ1alhN8XHhuqFFuGoEcTrYymTbkUyN/mxbyf+JG1rU0j9vjlXNYC637Gl2p+co4d01OplbqOjwIM2+rwOnd/ZwhcVO3f72z9pyOFMt64Toswc7ydfGzvlmICxMtvdErhb/sk/G9mP/O3uo1SK1bxCmXnER6hkTrlq8uVRYBUUOvb50j6av2CenIVUL8NEL/WLUOy6CtRQeLDUjR4Omr9WJHLua1grU7Ps68seCBxu3YLtmrj6g4Co2ffNoF0WF+bt6pAqDsChDB0/laVFyuhYlpWnrkaySr1ssUtu6VRUfF6H42HC2s1cgGw6c1tNzt2nfyXObx/q2itQ/+rTgr9sKYu+JXA2evkbHs+1qXDNQicM7qGYQfyR4mm+2pWlE4iZJ0rt3X6nuLWq5eKKKhbAoJ4dP52vJ9nNrMn49zPBXV0SFqldcuOJjI6hmD5VfWKxXFu/WBz8fkGFINYN89a/+cerBL6wK58DJPA2avkZpWQVqWCNAHw/vyBpID7LvRK76vvWTcu3FevDaRnomPtrVI1U4hIULpGWd1eLkdC1KStf6g6f123/ZuNohiv8lMhpU57wGnuCn1JN6Zt42HT597nDk26+so2d7t+DQxArs0Kl8DZq+Rkczz6p+NX99fH9HRYSw5tHdnS10qP9/ftKu9By1bxCmxPs6cNr8MkBYuFhGdsEvazLStXb/KTl/86/cPCJYvWLDFR8XwXHVbii7oEjjF+7Sx+sOSZJqh/pp/C1xuqZpDRdPhvJw+PS5uDhy5qzqhp2Li9ps1nRroz/bqs82HlH1QB8tfLQLO1KXEcLCjZzMteu7Hce1MClNq/eekuM3ldG0VqDiYyPUKy5CTWsFshOgiy3fdVx/m5es9OxzZ2a9q2M9jYmP5tTPlczRzLMa9M4aHTqdrzpV/fTx8I5sznRTn244rKfnbpPVIn00rIOualzd1SNVWISFmzqTV6jvdh7XoqQ0rUo9qSLH//75G1YPKNlcEhMZTGSUo8z8Qr341Q7N23xUklSvmr8m3tpSHRtWc/FkcJW0rHNxceBUvmqH+ilxeAdOz+5mdqZl6+apP8le7NRTNzTVyG5NXD1ShUZYeICss0VatvO4Fiala0XKCRUWO0uW1Q3zV3xcuHrFRqhlnRAiowwtSkrTc19u18lcuywWadjVDfTkDc24MB10PLtAg6av0b4TeYoIqaLE4R3ZR8pN5BQUqe9bP2n/yTxd16yG3r+nHafQL2OEhYfJKSjS8l0ZWpSUru93Z8j+m8ioHeqn+F/2yWgdFcoPj0lO5Nj1jwXJWpiULklqXDNQrwxoqTZ1q7p4MriTjJwCDZ6+VqkZuaoV7KvE4R3VqAb7RrmSYRgakbhJC5PSFRlSRd882oWzppYDwsKD5dmL9cPuE1qYnKbvd2Uov9BRsiw8uIpujA1Xr7gIta1XlUtxXwLDMLRg6zGNW7BdZ/KL5GW16KFrG+mR6xtzhj78oZO5diVMX6vdx3NUI8hXHw/voMY1g1w9VqU146f9euGrHfL2suiTBzrxx0A5ISwqiLOFDv2454QWJ6dp6c4M5dqLS5bVCPLVjTHhio8LV/v6YRxedRHSswr07BdJWrYrQ9K5I3QmDWip2NohLp4M7u5Url0J767VrvQcVQ/00ez7OqpZOHFR3jYfOqPb3/5ZRQ5Dz9/UQkM7N3D1SJUGYVEBFRQ59FPqSS1MStd3O9KVXfC/yAgL8FHPmFqKj41Qp0bV5E1knMcwDH264bD++fVO5diL5e1l0aPdmujB6xrxb4WLdiavUHe+t1bbj2UrLMBHs+/roOYR/B4rL2fyCtV78kodyypQr7hwTR3chv3PyhFhUcEVFju1eu9JLUpK15Id6crMLypZFuLnrRta1FKvuAhd3bi6fGyV+43z8Ol8jZ2XpFWpJyVJreqEaNJtrdS0Fn9tovSy8ot01/trte1IlkL9vfXRsA6s8SoHTqehoR+s1w+7T6hB9QAtGHm1gqpwsrryRFhUIkUOp9buO62FyWlakpyuU3mFJcuCqtjUo3ktxcdFqEuT6qriXXn2IXA6Dc1ac1ATF+9SfqFDvjarnryhqYZe3YDNRrgsWWeLdPf767T1cKZC/M7FRVwd4qIsvbU8Ra9+u0e+Nqvmj7iaNUUuQFhUUg6noXX7T2tRcpoWJafrRI69ZFmAj5eub15LveLCdW3TmhX6cMr9J/M0Zu42rTtwWpLUvn6YJg5oyaGCME12QZGGvL9Omw5lKqiKTbOGddAVUaGuHqtCWr33pO58d62chvTKrS11e7soV49UKREWkNNpaOOhM1qUlK5FyWlKyyooWebn7aVu0TUVHxeurs1qKqCCnFnS4TT03qp9eu3bPbIXO+Xv46Vn4qN1Z4d6HKYL0+Xai3XvjHVaf+CMgnxtmjm0vdrW4wgFM2VkF6jX5FU6mWvXgLZ19OptrVw9UqVFWOA8TqehrUcytSj53JVYj5w5W7LM12bVtU1rqFdchLo1r6lgD91uued4jkbP3aathzMlSZ0bV9f4W+I4FTPKVJ69WENnrtfa/acV4OOlmUPbq139MFePVSEUO5wa/O5ardt/WtHhQfri4asr9JpWd0dY4IIMw1Dy0WwtTE7TwqQ0HTyVX7LMx8uqLk2qKz4uQj2a11KIv/tHRpHDqf/+sFeTl6eoyGEoyNemv9/UXLdfGcUe4ygX+YXFuu+DDVq995T8fbz0/pB2nA7eBBMX79K0H/Yq0NemBSOvVkNOTOZShAUuimEY2pmWo0XJafomKU37TuSVLLNZLbq6cXX1igtXjxbhCnPDM9slH83S6LnbtDMtW5J0fXRN/at/nMJDuLohytfZQofun7VBK1NOqoq3Ve/f044LYl2GZTuPa9gHGyRJUwe3Ue+WES6eCIQFSs0wDKVk5GphUpoWJaVr9/GckmVeVos6Naym+Lhw3dAiXDWCfF04qWQvdmjKslRN+3GvHE5Dof7eeqFvjPq2imQtBVymoMihB2Zt1I97TsjXZtW791ypLk1quHosj3P4dL5umrJKWWeLNOSq+hrXN8bVI0GEBUyw90SuFv+yT8b2Y9klX7dapPYNwtQrLkI9Y8JVK7h81w5sOnRGT8/dptSMXElS77gIjesb4/LYAaRz0fvQR5u0fFeGfGxWvXNXW13XrKarx/IY9mKHbvvvz9p2JEutokL12QOdKv25eNwFYQFTHTyVp0XJ6VqUlKatR7JKvm6xSG3rVlV8XITiY8MVGepXZjOcLXTotW93672f9sswpOqBvnqpX4zi41hFCvdiL3ZoZOJmfbfjuHy8rPrvXW3ULbqWq8fyCM9/mawPfz6oUH9vff1IZ9Wpys7X7oKwQJk5fDpfS7afW5Ox6VDmecuuiApVr7hwxcdGmHo0xpp9pzTm820lO5re0rq2nrupBVc0hNsqLHbq0Y83a/H2dHl7WfSfhLbq0YK4+DMLth7Tox9vliTNuLedurKmx60QFigXaVlntTg5XYuS0rX+4Gn99n9TXO0QxceFq1dshOpf4ompcu3FmrBopz5ac0iSFBFSRS/3j1PXaH7hwP0VOZx6/JMt+mZbmmxWi94a3EY3xoa7eiy3lJqRq75vrVJ+oUMjujbS6J7Rrh4J/0+ZhMX48eM1b9487dq1S35+frrqqqs0ceJENWvWzPTB4Hkysgt+WZORrrX7T8n5m/9ZzSOC1Ss2XPFxEWpc8+IOGVux54TGzkvS0cxz59wY1L6uxvaK9tjzbKByKnY4NerTrVqw9Zi8rBZNHtiaIxz+n/zCYt089SftOZ6rjg3D9NGwDpx23w2VSVjceOONGjhwoNq1a6fi4mL97W9/U3Jysnbs2KGAgIv7i5SwqBxO5tr13Y7jWpiUptV7T8nxm8poWitQ8bER6hUXoaa1An93FEdWfpH++c0OfbbxiCSpTlU/Tby1pa7m0D14KIfT0OjPtmre5qPyslr0+h1XqG+rSFeP5RYMw9CTn23VvE1HVSPIV9882lk1gzhc3B2Vy6aQEydOqGbNmvrxxx91zTXXmDoYKo4zeYX6budxLUpK06rUkypy/O+/XMPqAYr/ZZ+MmMhgLd2ZoWe/SFJGjl0Wi3RPp/oa3bNZhTnlOCovh9PQmM+3ae7GI7JapNdub6X+reu4eiyXm7PukJ6ZlySrRUoc3pETi7mxi33/vqzf1llZ544OCAu78Olr7Xa77Pb/XQgrOzv7grdFxVQ1wEe3Xxml26+MUtbZIi3beVwLk9K1IuWE9p3M09Tv92rq93tVI8i35KJpDasHaOKAlpwaGRWGl9WiV25tKZvVojnrD2vUp1vlcEoD2lbeuNh+LEvPL9guSXqqZzOiooK45DUWTqdTffv2VWZmplatWnXB240bN04vvPDC777OGgvkFBRp+a4MLUpK1/e7M2QvdspqkYZf01BPdG9aqS7xjsrD6TT03JfJmr32kCwWacItcbqjXV1Xj1XusguK1GfKKh08la/ro2tq+t1XcqFAN1fmm0IeeughLVq0SKtWrVKdOhcu7j9aYxEVFUVY4Dx59mKtO3BaUVX9L3rnTsBTGYahcQu264OfD0qS/tU/Vgkd6rl4qvJjGIYe/Gijlmw/rtqhfvrm0c4K9efQcXdXpptCRo4cqa+//lorVqz406iQJF9fX/n6ckZE/LkAXxvHrKPSsFgsGtc3Rl5Wq97/ab+e/SJZDqehuzvVd/Vo5eK9Vfu1ZPtxeXtZNDWhDVFRwZTqeB7DMDRy5Eh98cUXWr58uRo0aFBWcwFAhWaxWPTcTc11/zUNJUnPf7ld76/a7+Kpyt7Gg6c1YdEuSdJzN7XQFVGhrh0IpivVGosRI0YoMTFRX375pYKCgpSeni5JCgkJkZ9f2Z3KGQAqIovForHx0fKyWjTth7168esdchqG7uvS0NWjlYlTuXaNTNysYqehPq0idVfHyrP5pzIp1T4WF7pq5IwZMzRkyJCLug8ONwWA8xmGode/26PJy1MlSWNujNZD1zVy8VTmcjgNDZmxTitTTqphjQAtGNlZgRxG7lHKZB+Lcj77NwBUChaLRaNuaCYvq1WvL92jiYt3yeF0amS3Jq4ezTRvLU/VypSTquJt1bSEtkRFBcY5UwHATTzWvYmeuqGpJOnVb/fojaV7XDyROValnNQby849l3/eHKdm4UEunghlibAAADcyslsTjbnx3AW43liaote+3e3Ra4vTswr02JzNMgxpYLuoSn1CsMqCsAAAN/PQdY30997NJUlTlqfqlSWeGRdFDqdGJm7SqbxCtYgI1ri+Ma4eCeWAsAAAN3Rfl4b6R58WkqRpP+zV+EW7PC4uJi3ZrQ0HzyjI16b/JLThbLqVBGEBAG7q3qsb6MV+5/7Kf2fFPr309U6PiYtvt6frnRX7JEmTbmup+tUv7grY8HyEBQC4sbs71de/+sdKkt7/ab/GLdju9nFx6FS+nvxsqyRp6NUNdGNshIsnQnkiLADAzSV0qKeJt8bJYpE++Pmg/j4/WU6ne8ZFQZFDDyduVE5BsdrUDdUz8dGuHgnljLAAAA9wR7u6mjSglSwWafbaQ/rbF0luGRcvfr1DyUezVdXfW28NbiMfG28zlQ2vOAB4iAFt6+jft7eS1SLNWX9YT3++TQ43iov5m48q8ZfLwb8xsLUiQ7nUQ2VEWACAB+nfuo7eGNhaXlaL5m48otGfbXWLuEg5nqOx85IkSY90baxrm9Zw8URwFcICADxM31aRmjKotWxWi+ZtPqonPtmiYofTZfPk2Yv10OxNOlvk0NWNq+mx7k1dNgtcj7AAAA/UKy5Cbw1uI5vVogVbj+mxOVtU5IK4MAxDz36RpNSMXNUK9tWbv6xNQeVFWACAh7oxNlzT7mwrby+LvklK0yOJm1VYXL5xMXvtIc3fckxeVoveGtxG1QN9y/Xx4X4ICwDwYD1a1NLbd7WVj5dVi7ena0TiJtmLHeXy2ElHsvTiVzskSU/3bKZ29cPK5XHh3ggLAPBw3aJrafo9V8rHZtV3O47roY82qaCobOMiK79IDyduVKHDqe7Na+n+axqW6ePBcxAWAFABXNu0ht6/p518bVYt35WhB2ZtLLO4MAxDT83dqsOnz6pOVT+9dlsrWSzsV4FzCAsAqCA6N6muGUPayc/bSz/uOaHhH24ok7iYvnKfvttxXD5eVk1LaKsQf2/THwOei7AAgArkqsbVNfPedvL38dLKlJMaOnO98guLTbv/dftPa+Li3ZKk5/u0UFydENPuGxUDYQEAFUyHhtX04dD2CvDx0uq9p3TvjPXKs19+XJzMteuRjzfJ4TTU74pIJXSoa8K0qGgICwCogK6sH6YPh3VQkK9Na/ef1pAZ65R7GXHhcBp6bM5mHc+2q3HNQL3cP479KvCHCAsAqKDa1quqWfd1UHAVm9YfOKO731urnIKiS7qvN5el6KfUU/Lz9tK0hDYK8LWZPC0qCsICACqwK6JCNfu+jgrx89amQ5m68711yjpburj4cc8JTVmeIkkaf0ucmtQKKotRUUEQFgBQwcXVCVHi8A6q6u+trYczdee7a5WZX3hR33ss86wen7NZhiEN7lBXN7euXcbTwtMRFgBQCcREhihxeEeFBfgo6WiWEt5dqzN5fx4XRQ6nRiZu0pn8IsXWDtbzN7Uop2nhyQgLAKgkmkcE6+PhHVU90Efbj2Vr0PQ1OpVrv+DtJyzapU2HMhVUxab/DG6rKt5e5TgtPBVhAQCVSLPwIM25v6NqBPlqV3qOBk1foxM5v4+Lxclpem/VfknSq7e1Ut1q/uU9KjxUqcNixYoV6tOnjyIjI2WxWDR//vwyGAsAUFYa1zwXF7WCfbXneK4GTV+jjJyCkuUHTuZp9GfbJEn3X9NQPWPCXTUqPFCpwyIvL0+tWrXS1KlTy2IeAEA5aFQjUJ/c30kRIVWUmpGrge+s0fHsAhUUOfTQ7E3KsRerXf2qGt2zmatHhYcp9YHI8fHxio+PL4tZAADlqH71AH1yfycNmr5G+07kaeA7axQTGaydadmqFuCjKYPayNuLLeYonTL/H2O325WdnX3eBwDAPdSt5q8593dUnap+2n8yT19vS5PFIr05sLXCQ6q4ejx4oDIPi/HjxyskJKTkIyoqqqwfEgBQClFh/vrkgU6qG3ZuB83Hrm+izk2qu3gqeCqLYRjGJX+zxaIvvvhCN9988wVvY7fbZbf/b4/j7OxsRUVFKSsrS8HBwZf60AAAk2UXFGlPeo7a1qvKdUDwO9nZ2QoJCfnL9+8yP9m7r6+vfH19y/phAACXKbiKt66sH+bqMeDh2CsHAACYptRrLHJzc5Wamlry+f79+7VlyxaFhYWpbt26pg4HAAA8S6nDYsOGDeratWvJ56NGjZIk3XPPPZo5c6ZpgwEAAM9T6rC47rrrdBn7ewIAgAqMfSwAAIBpCAsAAGAawgIAAJiGsAAAAKYhLAAAgGkICwAAYBrCAgAAmIawAAAApiEsAACAaQgLAABgGsICAACYhrAAAACmISwAAIBpCAsAAGAawgIAAJiGsAAAAKYhLAAAgGkICwAAYBrCAgAAmIawAAAApiEsAACAaQgLAABgGsICAACYhrAAAACmISwAAIBpCAsAAGAawgIAAJiGsAAAAKYhLAAAgGls5f2AhmFIkrKzs8v7oQEAwCX69X371/fxCyn3sMjJyZEkRUVFlfdDAwCAy5STk6OQkJALLrcYf5UeJnM6nTp27JiCgoJksVhMu9/s7GxFRUXp8OHDCg4ONu1+UX54DT0fr6Fn4/XzfGX5GhqGoZycHEVGRspqvfCeFOW+xsJqtapOnTpldv/BwcH8QHg4XkPPx2vo2Xj9PF9ZvYZ/tqbiV+y8CQAATENYAAAA01SYsPD19dU//vEP+fr6unoUXCJeQ8/Ha+jZeP08nzu8huW+8yYAAKi4KswaCwAA4HqEBQAAMA1hAQAATFMhwmLIkCG6+eabXT0GAACVXrmHxZAhQ2SxWGSxWOTj46PGjRvrxRdfVHFx8V9+74EDB2SxWLRly5ayHxTlKj09XY888ogaNmwoX19fRUVFqU+fPlq2bJmrR6u0eE08H390VQx/9r75ww8/yGKxKDMz83ffV79+fb3xxhvnff7r/QQEBKhNmzb67LPPTJ/XJWssbrzxRqWlpSklJUVPPvmkxo0bp0mTJrliFLiBAwcOqG3btlq+fLkmTZqkpKQkLV68WF27dtWIESNcPV6ldCmvicVi0YEDBy7q/mfOnKnrrrvOvIGBCs6s980XX3xRaWlp2rx5s9q1a6c77rhDq1evNnVWl4SFr6+vwsPDVa9ePT300EPq3r27Pv30UwUHB2vu3Lnn3Xb+/PkKCAhQTk6OGjRoIElq3bq1LBbL734xvfrqq4qIiFC1atU0YsQIFRUVlSw7c+aM7r77blWtWlX+/v6Kj49XSkpKyfKZM2cqNDRUS5YsUfPmzRUYGFjyQqJsPfzww7JYLFq3bp1uvfVWNW3aVDExMRo1apTWrFnj6vEqJV6Timfx4sXq3LmzQkNDVa1aNd10003au3dvyfKrrrpKY8aMOe97Tpw4IW9vb61YsUKSNGvWLF155ZUKCgpSeHi4Bg8erIyMjHJ9HpXVH71vLliwoNT38+tr17RpU02dOlV+fn766quvTJ3VLfax8PPzk9Vq1cCBAzVjxozzls2YMUMDBgxQUFCQ1q1bJ0launSp0tLSNG/evJLbff/999q7d6++//57ffDBB5o5c6ZmzpxZsnzIkCHasGGDFixYoJ9//lmGYahXr17nxUd+fr5effVVzZo1SytWrNChQ4f01FNPle2Tr+ROnz6txYsXa8SIEQoICPjd8tDQ0PIfqpLjNamY8vLyNGrUKG3YsEHLli2T1WpV//795XQ6JUkJCQmaM2fOeZfE/uSTTxQZGakuXbpIkoqKivTSSy9p69atmj9/vg4cOKAhQ4a44ulUen5+fiosLLys+7DZbPL29r7s+/nd/Zp6b6VkGIaWLVumJUuW6JFHHtFtt92mq666SmlpaYqIiFBGRoYWLlyopUuXSpJq1KghSapWrZrCw8PPu6+qVavqrbfekpeXl6Kjo9W7d28tW7ZMw4cPV0pKihYsWKCffvpJV111lSRp9uzZioqK0vz583XbbbdJOvdD89///leNGjWSJI0cOVIvvvhief1zVEqpqakyDEPR0dGuHgW/4DWpmG699dbzPn///fdVo0YN7dixQ7Gxsbr99tv1+OOPa9WqVSUhkZiYqEGDBpVciXro0KEl39+wYUNNnjxZ7dq1U25urgIDA8vvyVRi//9981d/dHHP/Pz8C95PYWGhXnvtNWVlZalbt26mzuiSNRZff/21AgMDVaVKFcXHx+uOO+7QuHHj1L59e8XExOiDDz6QJH300UeqV6+errnmmr+8z5iYGHl5eZV8/muYSNLOnTtls9nUoUOHkuXVqlVTs2bNtHPnzpKv+fv7l0TF/78PlA1O/Op+LvY1iY+PV2BgYMmHdO7n8NfPY2JiSm576NCh82774IMPauXKled97eWXXy6T54NzUlJSNGjQIDVs2FDBwcGqX7++pHOvjXTuD7cbbrhBs2fPliTt379fP//8sxISEkruY+PGjerTp4/q1q2roKAgXXvttefdB8rOhd43f7Vy5Upt2bLlvI/IyMjf3c+YMWMUGBgof39/TZw4URMmTFDv3r1NndUlayy6du2qadOmycfHR5GRkbLZ/jfGfffdp6lTp+qZZ57RjBkzdO+995bU8p/x9vY+73OLxVKyiu9i/dF98MZXtpo0aSKLxaJdu3a5ehT84mJfk3fffVdnz5497/sWLlyo2rVrSzr/5ykyMvK8o7nmzZunzz//vORNTJLCwsJMegb4I3369FG9evU0ffp0RUZGyul0KjY29rzV4AkJCXr00Uc1ZcoUJSYmKi4uTnFxcZLObUrp2bOnevbsqdmzZ6tGjRo6dOiQevbsafqqdPzen71vSlKDBg1+t5ny/99GkkaPHq0hQ4YoMDBQtWrVuqj319JySVgEBASocePGf7jszjvv1NNPP63Jkydrx44duueee0qW+fj4SJIcDkepHq958+YqLi7W2rVrSzaFnDp1Srt371aLFi0u8VnADGFhYerZs6emTp2qRx999Hfb9DMzM9mmX84u9jX5NSB+q169eiV/Cf+WzWY772e+Zs2a8vPzu+DvAZjr199306dPL9nMsWrVqt/drl+/frr//vu1ePFiJSYm6u677y5ZtmvXLp06dUoTJkxQVFSUJGnDhg3l8wTwp++bpVG9evUy/7lzi503f6tq1aq65ZZbNHr0aN1www3nbTf69ZfR4sWLdfz4cWVlZV3UfTZp0kT9+vXT8OHDtWrVKm3dulV33nmnateurX79+pXVU8FFmjp1qhwOh9q3b6/PP/9cKSkp2rlzpyZPnqxOnTq5erxKidekYqlataqqVaumd955R6mpqVq+fLlGjRr1u9sFBATo5ptv1nPPPaedO3dq0KBBJcvq1q0rHx8fTZkyRfv27dOCBQv00ksvlefTgIdwu7CQpGHDhqmwsPC8HYWkc3/1TJ48WW+//bYiIyNLFQUzZsxQ27ZtddNNN6lTp04yDEMLFy783eYPlL+GDRtq06ZN6tq1q5588knFxsaqR48eWrZsmaZNm+bq8SolXpOKwel0ymazyWq1as6cOdq4caNiY2P1xBNPXPAcCAkJCdq6dau6dOmiunXrlny9Ro0amjlzpj777DO1aNFCEyZM0KuvvlpeTwUexC0vmz5r1iw98cQTOnbsWMnmDwBA6dx4441q3Lix3nrrLVePgkrErdZY5Ofna+/evZowYYIeeOABogIALsGZM2f09ddf64cfflD37t1dPQ4qGbcKi1deeUXR0dEKDw/X2LFjXT0OAHikoUOH6sEHH9STTz7JfmQod265KQQAAHgmt1pjAQAAPBthAQAATENYAAAA0xAWAADANIQFAAAwDWEBAABMQ1gAAADTEBYAAMA0hAUAADDN/wFVQ1jqYdj1wwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "x=[1,2,3,4,5]\n",
    "y=[3,2,4,1,5]\n",
    "\n",
    "plt.plot(x,y)\n",
    "plt.xticks(x, labels=[\"Python\", \"C\",\"C++\",\"Java\", \"PHP\"])\n",
    "#plt.xlim(0,10)\n",
    "#plt.ylim(0,10)\n",
    "plt.yticks(y)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "deb99207-06ec-4e77-a053-ce82e6b5d81d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}