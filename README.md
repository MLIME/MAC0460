# MAC0460 - Introdução ao aprendizado de máquina
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/MLIME/MAC0460/blob/master/LICENSE)

Nesse repositório estão os diferentes materais da disciplina **MAC0460 - Introdução ao aprendizado de máquina** ministrada no Instituto de Matemática e Estatística (IME) da Universidade de São Paulo (USP). Maiores informações podem ser encontradas [aqui](https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=MAC0460). 

Um pedaço da parte teórica do curso foi baseado no curso [**Learning from Data**](https://work.caltech.edu/telecourse.html), vale a pena conferir esse material.

## Uso

Na pasta **notebooks** estão os exercícios práticos dados no curso, na pasta **slides** estão os materiais didáticos de algumas aulas.

### Instalação (Ubuntu / Debian)
Para instalar o [Jupyter Notebook](http://jupyter.org/) basta rodar:

```
$ sudo apt install python3-pip python3-tk
$ pip3 install --upgrade pip3
$ sudo pip3 install jupyter
```

Vamos usar uma série de bibliotecas de Python ao longo da disciplina, para instalar as principais rode:

```
$ pip3 install -r requirements.txt
```

É preciso ir no site do [PyTorch](https://pytorch.org/) para instalar essa biblioteca.

Para acessar os notebooks basta rodar:

```
$ cd notebooks
$ jupyter notebook
```

### Pontos importantes

- Usamos a biblioteca de deep learning PyTorch. Essa biblioteca esta mudando constantemente. Os exercícios práticos foram desenvolvidos para a versão 0.4.0. Não é garantido que os notebooks vão funcionar para as versões futuras.

- Partimos do pressuposto que o usuário está usando Ubuntu ou Debian. A compatibilidade com outros sistemas não foi testada.

- Os notebooks foram feitos para a versão 3.5 do Python


### Referências

Vale a pena se familiarizar com cada uma das bibliotecas que vão ser usadas:
- [Jupyter](https://jupyter.readthedocs.io/en/latest/)
- [NumPy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
- [Matplotlib](https://matplotlib.org/tutorials/index.html)
- [PyTorch](https://pytorch.org/tutorials/)



