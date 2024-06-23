{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "00fd1322-a85f-4982-a66f-ca7529a17aff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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2qWXX37ZPm7Dhg369NNPtWHDBi1fvlx1dXWqq6uz98+ePVtbtmzRK6+8osbGRlmWpUsvvTQkYnp7e/Xwww/r+eef16ZNm7Rz507dcccd4f3lATiKQAFwzCzL0ptvvql169ZpypQpuvHGG7Vu3Tr7rEVXV5dee+013XDDDZKkk046SZKUlZUln8+nzMxMe60RI0boiSeeUEFBgaZPn67LLrtMDQ0NkqTW1la98sor+tOf/qSf/OQnOuuss7Ry5Up98cUXWrNmjb3GwMCAampqNGHCBI0fP15z58611wAQHQgUAEdt7dq1SktLU1JSkqZNm6af//znuvfee3XOOefotNNO0/LlyyVJK1as0OjRo3XBBRf84JqnnXaaYmNj7fs5OTnq6uqSJLW0tCguLk4TJ06092dlZenHP/6xWlpa7G0pKSk65ZRTDrsGgOhAoAA4apMnT9Z7772n1tZWff3111q+fLlSU1MlSTfeeKP90kxtba2uv/56uVyuH1wzPj4+5L7L5dLg4OCQ5jrcGnyrBxBdCBQARy01NVWnnnqq8vLyFBcX+qbAWbNmaceOHaqurtaHH36o6667zt6XkJAgSTp48OCQfl5hYaEOHDigpqYme9uXX36pbdu2aezYscfwmwAwDYECICxGjBihq666SnfeeacuvvhijRo1yt6XnZ2t5ORk1dfXq7OzU4FA4IjWHDNmjK644grddNNNevvtt/X+++9r1qxZ+tGPfqQrrrgiXL8KgAggUACEzZw5c7R//3774thvxMXFqbq6Wk899ZT8fv+Q4qK2tlZFRUWaPn26iouLZVmWXnvtte+8rAMgurksXpgFECbPP/+85s+fr/b2dvtlHQA4EnySLADH9fb2ateuXXrwwQd18803EycAhoyXeAA4bvHixSooKJDP5+Nj7wEcFV7iAQAAxuEMCgAAMA6BAgAAjEOgAAAA4xAoAADAOAQKAAAwDoECAACMQ6AAAADjECgAAMA4BAoAADDO/wNwErzNLwgVDwAAAABJRU5ErkJggg==",
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
    "x =[10,20,30,40,50,60,70,120]\n",
    "\n",
    "plt.boxplot(x, tick_labels=[\"Python\"],patch_artist=True,showmeans=True,sym=\"g*\", boxprops=dict(color=\"r\"), capprops=dict(color=\"g\"))\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "cd9494b8-e42d-4192-8ba9-22bed47d55fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Vdu2bZMkbdq0Sfv27dNrr70WOO69997Tl19+qffee08vvPCCamtrVVtbG9i/cOFC7dixQ2+88YYaGxvl9/v185//PChQDh48qMcff1wvvfSStmzZovb2dt11112D+8UDPxD//ve/VVdXp9LSUo0cOfKY/WlpaeaHQkwjPhAWv9+vTZs2qb6+XjNmzNDNN9+s+vr6wKsN+/fv11tvvaWbbrpJkjR69GhJUkZGhhwOh9LT0wNrjRo1Sk8//bScTqdmzpypq6++OvAecltbm9544w09//zzuuyyy3T++efr5Zdf1ldffaUNGzYE1ujr69OqVat04YUXavLkybr99tt5HxqIki+++EJ+v19Op3OoR8EwQXwgJG+++aZSUlKUlJSk4uJiXXfddXrggQd08cUXa+LEiXrhhRckSWvWrNG4ceN0+eWXf++aEydO1GmnnRa4n5WVpf3790uSWlpaFB8frylTpgT2Z2Rk6JxzzlFLS0tg24gRI/SjH/3ouGsAiMxAfxF2cXGxUlJSAjfpyPl99P7EiRMDx7a3twcde+utt+qDDz4I2vbwww8PyteDoRc/1AMgtkyfPl3V1dVKTExUdna24uP/93+hm2++WVVVVfrd736nmpoa/fKXv5TFYvneNRMSEoLuWywW9ff3hzTX8dbgkwOA6JgwYYIsFot27dp10uOef/55HTp0KOhxb731ls444wxJwedpdnZ20E+/vfbaa3r11Vf18ssvB7b95yukGF6ID4Rk5MiROuuss467b/78+br77ru1YsUKffbZZ1qwYEFgX2JioiTp8OHDIf19+fn5+u6777R161ZdcsklkqSvv/5ara2tOvfcc8P8KgCEIj09XUVFRaqqqtIdd9xxzHUfnZ2dSktLC0TGfxo3bpzGjx9/zPb4+Pig7yVjxoxRcnLyCb+/YHjhbRdEzahRozR37lz99re/1VVXXaWxY8cG9h39xlJXVyePx6Ourq4BrTlhwgTNnj1bt9xyiz788EP99a9/1fz583XGGWdo9uzZg/WlAPgvVVVVOnz4sC6++GK9+uqramtrU0tLi1asWCGXyzXU4yHGEB+IqkWLFqm3tzdwoelR8fHxWrFihZ555hllZ2eHFA41NTUqKCjQzJkz5XK55Pf79dZbbx3zVguAwXPmmWfq448/1vTp03XnnXfqvPPO05VXXqmGhgZVV1cP9XiIMRY/b4wjil566SWVlZWpo6Mj8FYLAAD/iWs+EBUHDx7Uvn379Mgjj+hXv/oV4QEAOCHedkFUPProo3I6nXI4HKqoqBjqcQAApzDedgEAAEbxygcAADCK+AAAAEYRHwAAwCjiAwAAGEV8AAAAo4gPAABgFPEBAACMIj4AAIBRxAcAADDq/wEIDpxJqiWm/QAAAABJRU5ErkJggg==",
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
    "x =[10,20,30,40,50,60,70]\n",
    "y=[10,20,30,40,50,60,70]\n",
    "z=[x,y]\n",
    "plt.boxplot(z, tick_labels=[\"Python\",\"C++\"],showmeans=True,sym=\"g*\", boxprops=dict(color=\"r\"), capprops=dict(color=\"g\"))\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed4325f4-ada1-4573-b84a-e5722354d541",
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