from sys import stdout

colors = {
    'clear': "\033[39m", # default colors
	'k': "\033[30m", # black
	'r': "\033[31m", # red
	'g': "\033[32m", # green
	'y': "\033[33m", # yellow
	'b': "\033[34m", # blue
	'm': "\033[35m", # magenta
	'c': "\033[36m", # cyan
    'w': "\033[97m" # white
}

def printc(c, s, end='\n', outfile=stdout, flush=False):
    """ Quase identica a print, imprime s (str) com a cor dada
    por c (str) dentre as opcoes:

        - 'r' (red/vermelho)
        - 'g' (green/verde)
        - 'y' (yellow/amarelo)
        - 'b' (blue/azul)
        - 'm' (magenta)
        - 'c' (cyan/ciano)
        - 'k' (black/preto)
        - 'w' (white/branco)

    As opcoes de cores sao dadas pelos caracteres especiais ANSI escape codes.
    """
    print(colors[c] + s + colors['clear'], 
            end=end, file=outfile, flush=flush)
