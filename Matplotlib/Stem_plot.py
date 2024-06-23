{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "06962ec5-7b8d-4317-b6a5-6e6086e312bb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhYAAAGdCAYAAABO2DpVAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAuyklEQVR4nO3dfXxU5Z338e+ZSTIkEIIgkUBCghblSVBEETEK9ancarFISxUtorXdNrhQ6t6a3fUJXaPr2kIrpaC9oU+U+kIo6i4qtTxpRZ5kRVEERQkQhFYlBMgkOefcf2SZNQImk5xzzeHM5/16zStnTubhx3dI5pfrXOcay3VdVwAAAB6IpLoAAAAQHjQWAADAMzQWAADAMzQWAADAMzQWAADAMzQWAADAMzQWAADAMzQWAADAMxmmn9BxHO3Zs0e5ubmyLMv00wMAgFZwXVcHDx5U9+7dFYmceFzCeGOxZ88eFRUVmX5aAADggcrKShUWFp7w+8Ybi9zcXEmNhXXs2NH00wMAgFaorq5WUVFR4n38RIw3FkcPf3Ts2JHGAgCAk0xz0xiYvAkAADxDYwEAADxDYwEAADxjfI4FAADH47quGhoaZNt2qktJS9FoVBkZGW1eCoLGAgCQcnV1daqqqtLhw4dTXUpay8nJUUFBgbKyslr9GDQWAICUchxHO3bsUDQaVffu3ZWVlcUCioa5rqu6ujrt379fO3bsUO/evb90EawvQ2MBAEipuro6OY6joqIi5eTkpLqctJWdna3MzEx99NFHqqurU7t27Vr1OEzeBAAEQmv/QoZ3vHgNGLEAkL5sW1q9WqqqkgoKpNJSKRpNdVVoLV7PQEi6Ndm9e7duuukmdenSRdnZ2Tr77LO1fv16P2oDAP8sWiSVlEgjR0o33tj4taSkcT9OPiF6PT/88ENZlqVNmzalupRWSaqx+PTTTzV8+HBlZmZq6dKl2rJlix5//HGdcsopftUHAN5btEgaO1batavp/t27G/efhG9Gae0kfj1vueUWXXfddakuw1NJHQp59NFHVVRUpLlz5yb29erVy/OiAMA3ti1Nniy57rHfc13JsqQpU6TRoxlGPxnwegZOUiMWzz77rIYMGaJvfvObys/P17nnnqsnn3zyS+8Tj8dVXV3d5AIAKbN69bF/2X6e60qVlY23Q/Cl+PUcMWKEJk2apEmTJikvL0+nnnqq7rnnHrmuq2nTpmnAgAHH3Oecc87RPffco/vvv1+//vWvtWTJElmWJcuytGLFisTtPvjgA40cOVI5OTkaNGiQXnvttSaP88wzz6h///6KxWIqKSnR448/3uT7JSUlevjhh3XrrbcqNzdXPXv21Jw5c3zJoQk3CbFYzI3FYm55ebm7ceNGd/bs2W67du3cefPmnfA+9913nyvpmMuBAweSeWoA8Mb8+a7b+Hbz5Zf581Ndado4cuSIu2XLFvfIkSPJ3znFr+ell17qdujQwZ08ebL77rvvur/73e/cnJwcd86cOW5lZaUbiUTctWvXJm6/ceNG17Is9/3333cPHjzofutb33K/9rWvuVVVVW5VVZUbj8fdHTt2uJLcPn36uM8//7y7detWd+zYsW5xcbFbX1/vuq7rrl+/3o1EIu60adPcrVu3unPnznWzs7PduXPnJp6ruLjY7dy5sztz5kx327ZtbkVFhRuJRNx33333hP+eL3stDhw40KL376Qai8zMTHfYsGFN9t1xxx3uhRdeeML71NbWugcOHEhcKisraSwApM7y5S17I1q+PNWVpo02NRYpfj0vvfRSt2/fvq7jOIl9d911l9u3b1/XdV131KhR7g9+8IPE9+644w53xIgRiesTJkxwR48e3eQxjzYWTz31VGLf22+/7Upy33nnHdd1XffGG290r7jiiib3+6d/+ie3X79+ievFxcXuTTfdlLjuOI6bn5/vzpo164T/Hi8ai6QOhRQUFKhfv35N9vXt21c7d+484X1isZg6duzY5AIAKVNaKhUWNh57Px7LkoqKGm+H4AvA63nhhRc2WSl02LBh2rZtm2zb1u23364//OEPqq2tVV1dnebPn69bb721RY87cODAxHZBQYEkad++fZKkd955R8OHD29y++HDhyee93iPYVmWunXrlngMvyTVWAwfPlxbt25tsu+9995TcXGxp0UBgG+iUWnGDEmS+8U3o6PXp09not/JIuCv57XXXqtYLKbFixfrueeeU319vcaOHdui+2ZmZia2jzYujuMk9fyff4yjj5PsYyQrqcbiRz/6kdasWaOHH35Y27dv1/z58zVnzhyVlZX5VR8AeG/MGGnhQql7j6b7Cwsb948Zk5q60Dr/83q6KXo9X3/99SbX16xZo969eyc+LXTChAmaO3eu5s6dq29/+9vKzs5O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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Stem Plot\n",
    "import matplotlib.pyplot as plt\n",
    "x =[1,2,3,4,5,6]\n",
    "y=[2,2,5,6,4,3]\n",
    "\n",
    "plt.stem(x,y,linefmt=\":\",markerfmt=\"ro\",bottom=0,basefmt=\"g\",label=\"python\")\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4c9fe00e-c201-439f-a2ff-f329d8004b0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Stem Plot\n",
    "import matplotlib.pyplot as plt\n",
    "x =[1,2,3,4,5,6]\n",
    "y=[2,2,5,6,4,3]\n",
    "\n",
    "plt.stem(x,y,linefmt=\":\",markerfmt=\"ro\",bottom=0,basefmt=\"g\",label=\"python\",orientation=\"horizontal\")\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59f23471-d655-42ba-b9e7-55a3b08f2329",
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
