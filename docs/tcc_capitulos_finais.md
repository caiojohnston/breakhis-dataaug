# Rascunho de Integração: Metodologia Atualizada, Resultados, Discussão e Conclusão

Este arquivo consolida o texto acadêmico a ser incorporado à versão completa do TCC. As referências listadas na seção "Referências verificadas no Consensus" foram buscadas no conector Consensus e tiveram o registro completo recuperado por `fetch` antes de serem citadas. As citações já presentes no documento principal podem ser mantidas, mas qualquer nova afirmação bibliográfica fora desta lista deve ser verificada antes da versão final.

Formulação central a preservar:

> A hipótese foi parcialmente confirmada e refinada: o sistema Encoder-Difusor melhorou o desempenho downstream em classificação binária quando as imagens sintéticas foram usadas em proporção controlada, embora não tenha validado plenamente o cenário de oito subtipos nem apresentado alta fidelidade gerativa.

## Ajustes Necessários na Metodologia

### Redefinição do protocolo experimental

O protocolo experimental foi inicialmente planejado para avaliar a classificação multiclasse dos oito subtipos histológicos do BreakHis. Esse desenho era coerente com o condicionamento do LDM, que foi treinado por subtipo, e com a hipótese original de que a equalização sintética poderia beneficiar principalmente as classes minoritárias. Contudo, a execução dos primeiros experimentos indicou que a formulação em oito subtipos era instável sob uma divisão estritamente patient-wise, em linha com avaliações recentes que tratam o split por paciente como requisito para evitar vazamento e estimativas otimistas em BreakHis [1].

Essa instabilidade não decorreu apenas do desbalanceamento em número de imagens, mas sobretudo do número reduzido de pacientes em alguns subtipos. No particionamento adotado, subtipos como phyllodes_tumor, adenosis, lobular_carcinoma e papillary_carcinoma ficaram representados por poucos pacientes em treinamento, validação e teste. O caso mais crítico foi phyllodes_tumor, com apenas três pacientes no total, distribuídos como um paciente em treino, um em validação e um em teste. Como as imagens de um mesmo paciente são altamente correlacionadas, a unidade estatística relevante para generalização é o paciente, não a imagem individual. Assim, a manutenção de um split por paciente, embora torne a tarefa mais difícil, foi preservada por ser metodologicamente mais rigorosa e por reduzir o risco de data leakage.

Diante desse diagnóstico, o protocolo downstream principal foi redefinido para classificação binária benigno versus maligno. A mudança não elimina o valor do condicionamento por subtipo no LDM: as imagens sintéticas continuaram sendo geradas por subtipo histológico, preservando granularidade no processo generativo. O que foi ajustado foi o endpoint de avaliação downstream, que passou a refletir uma tarefa mais robusta estatisticamente no BreakHis sob split patient-wise.

### Organização dos cenários comparativos

Foram mantidos três cenários experimentais principais:

| Cenário | Composição do treinamento | Finalidade |
|---|---|---|
| A | Imagens reais do BreakHis | Linha de base sem aumento de dados |
| B | Imagens reais com aumento clássico online | Controle com transformações geométricas e fotométricas |
| C | Imagens reais combinadas com sintéticas geradas pelo LDM | Avaliação do sistema Encoder-Difusor proposto |

No protocolo original de oito subtipos, o Cenário C utilizava a equalização completa por subtipo. Na versão revisada do estudo, esse caso passa a ser denominado C100, pois usa 100% das imagens sintéticas geradas para balanceamento. A partir dos resultados preliminares, foram conduzidas ablações adicionais para avaliar a proporção de dados sintéticos incorporada ao treino:

| Variante | Proporção de sintéticas usadas | Total de treino informado |
|---|---:|---:|
| C100 | 100% das sintéticas geradas | Reais + equalização completa |
| C25 | 25% das sintéticas geradas | 4.892 reais + 3.082 sintéticas |
| C50_full | 50% das sintéticas geradas | 4.892 reais + 6.164 sintéticas |

Essa ablação foi necessária porque a hipótese experimental refinada deixou de ser "quanto mais sintético, melhor" e passou a ser "dados sintéticos por difusão podem melhorar o downstream quando usados como augmentação controlada". Essa leitura é compatível com revisões sobre aumento generativo em imagem médica e com estudos de LDM em classificadores médicos, nos quais os ganhos dependem da tarefa e da proporção real-sintético [2,3]. Portanto, a proporção de sintéticas tornou-se uma variável metodológica central.

### Calibração de threshold

Na classificação binária, além da decisão padrão por argmax, foi avaliada a calibração do limiar de decisão com base no conjunto de validação. O limiar foi escolhido exclusivamente na validação, maximizando balanced accuracy, e depois aplicado uma única vez ao conjunto de teste. Esse procedimento foi adotado porque os modelos generativos apresentaram melhora consistente em AUC, indicando melhor ordenação probabilística, mas nem sempre a decisão padrão em 0,5 produzia o melhor equilíbrio entre sensibilidade e especificidade.

O principal resultado downstream completo é o C50_full calibrado, com threshold 0,97 definido na validação. O C25 permanece relevante por apresentar a maior AUC entre os experimentos completos, mas o C50_full calibrado foi selecionado como resultado principal por reunir os maiores valores completos de acurácia, balanced accuracy e F1 macro.

### Avaliação estatística

Para comparar o baseline binário A com o melhor modelo completo C50_full calibrado, foram aplicadas duas análises complementares. Primeiro, o teste exato de McNemar foi calculado por imagem, considerando os acertos e erros pareados dos dois classificadores no conjunto de teste. Segundo, foi aplicado bootstrap por paciente, de modo a respeitar a estrutura patient-wise do BreakHis e a dependência entre imagens de um mesmo indivíduo.

Essa dupla leitura é importante: a análise por imagem fornece poder estatístico maior, mas pode superestimar a independência das amostras; a análise por paciente é mais conservadora e mais alinhada à unidade real de generalização, porém sofre com o tamanho reduzido do conjunto de teste, composto por apenas 17 pacientes.

### Limitações metodológicas incorporadas

As seguintes limitações devem aparecer explicitamente no Capítulo 3 e ser retomadas na Discussão:

- O VAE não foi fine-tunado no domínio BreakHis por restrição de hardware. O modelo `stabilityai/sd-vae-ft-mse` foi usado congelado.
- Não foi implementado um baseline GAN no mesmo ambiente experimental; a comparação com GANs permanece conceitual e bibliográfica.
- Não houve avaliação qualitativa das imagens sintéticas por patologista.
- A classificação em oito subtipos mostrou-se instável sob split patient-wise devido ao baixo número de pacientes em subtipos raros.
- As métricas FID, SSIM, LPIPS e PSNR indicaram qualidade gerativa limitada.
- O conjunto de teste possui apenas 17 pacientes, de modo que inferências estatísticas por paciente devem ser interpretadas com cautela.

## 4 Resultados

Este capítulo apresenta os resultados experimentais do sistema híbrido Encoder-Difusor aplicado ao aumento de dados em imagens histopatológicas do BreakHis. A exposição segue a evolução real do estudo: primeiro, são apresentados os resultados do protocolo inicial de oito subtipos, que funcionaram como diagnóstico das limitações do desenho original; em seguida, são apresentados os experimentos binários benigno versus maligno, a ablação de proporção sintética, a calibração de threshold, os testes estatísticos e as métricas de qualidade gerativa.

### 4.1 Diagnóstico inicial: classificação em oito subtipos

A formulação original do experimento utilizou os oito subtipos histológicos do BreakHis como classes de saída. Essa escolha era metodologicamente compatível com o LDM condicionado por subtipo, mas mostrou-se instável quando combinada com o split por paciente. A Tabela 1 sintetiza os resultados obtidos nos três cenários originais.

**Tabela 1 - Resultados do protocolo inicial com oito subtipos**

| Cenário | Descrição | Accuracy | F1 macro |
|---|---|---:|---:|
| A | BreakHis original | 49,83% | 27,71% |
| B | BreakHis + aumento clássico | 41,52% | 22,26% |
| C100 | BreakHis + sintéticas LDM | 43,98% | 24,95% |

Os resultados não sustentaram a hipótese forte C > B > A no cenário multiclasse. O aumento clássico reduziu o desempenho global em relação ao baseline, e o aumento generativo recuperou parte dessa perda, mas permaneceu abaixo do Cenário A em acurácia e F1 macro. A análise das matrizes de confusão e das métricas por subtipo indicou que o classificador tendia a favorecer a classe majoritária, ductal_carcinoma, enquanto subtipos raros como phyllodes_tumor e lobular_carcinoma eram pouco reconhecidos.

Esse comportamento deve ser interpretado à luz da estrutura patient-wise do conjunto. Embora o BreakHis possua 7.909 imagens, a base contém apenas 81 pacientes. Em vários subtipos, há menos de dez pacientes no total; em phyllodes_tumor, há somente três. Assim, a tarefa de oito subtipos, sob split rigoroso por paciente, depende de generalização a pacientes nunca vistos em classes com representação extremamente pequena. A baixa macro F1, portanto, não indica apenas deficiência do classificador, mas uma limitação estatística da formulação multiclasse nesse protocolo.

O principal achado dessa etapa foi negativo, mas metodologicamente útil: a hipótese original precisava ser refinada. A classificação em oito subtipos foi mantida como diagnóstico exploratório, enquanto o endpoint principal de avaliação downstream foi redefinido para a classificação binária benigno versus maligno.

### 4.2 Resultados binários: baseline, aumento clássico e C100

Na formulação binária, o pipeline apresentou desempenho substancialmente mais estável. A Tabela 2 compara os cenários A, B e C100 no conjunto de teste, mantendo split por paciente e sem acesso ao teste durante treinamento ou calibração.

**Tabela 2 - Resultados binários iniciais**

| Cenário | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|
| A | 85,89% | 81,13% | 82,88% | 89,79% |
| B | 84,30% | 81,27% | 81,80% | 90,36% |
| C100 | 84,90% | 82,24% | 82,60% | 91,14% |

O baseline A atingiu 85,89% de acurácia e F1 macro de 82,88%, demonstrando que a tarefa binária é mais adequada à estrutura estatística do conjunto. O aumento clássico apresentou efeito misto: melhorou discretamente balanced accuracy e AUC em relação a A, mas reduziu acurácia e F1 macro. O C100, por sua vez, superou B em todas as métricas e superou A em balanced accuracy e AUC, embora ainda ficasse ligeiramente abaixo do baseline em acurácia e F1 macro.

Essa diferença entre AUC e métricas baseadas em threshold sugeriu que o modelo treinado com sintéticas apresentava melhor ordenação probabilística das amostras, mas não necessariamente a melhor fronteira de decisão no limiar padrão. Por esse motivo, foram conduzidas duas extensões: calibração de threshold por validação e ablação da proporção de imagens sintéticas.

### 4.3 Ablação da proporção de imagens sintéticas

A ablação C25/C50_full/C100 avaliou se o uso controlado das imagens sintéticas seria mais efetivo do que a equalização completa. A Tabela 3 apresenta o comparativo principal dos experimentos completos e suas versões calibradas quando aplicável.

**Tabela 3 - Comparativo binário com ablação de sintéticas**

| Variante | Decisão | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---|---:|---:|---:|---:|---:|
| A | Argmax | - | 85,89% | 81,13% | 82,88% | 89,79% |
| B | Argmax | - | 84,30% | 81,27% | 81,80% | 90,36% |
| C100 | Argmax | - | 84,90% | 82,24% | 82,60% | 91,14% |
| C25 | Argmax | - | 87,56% | 84,16% | 85,31% | 94,05% |
| C50_full | Argmax | - | 87,29% | 83,81% | 84,98% | 92,63% |
| C25 calibrado | Validação | 0,79 | 87,89% | 85,20% | 85,91% | 93,76% |
| C50_full calibrado | Validação | 0,97 | 88,02% | 86,73% | 86,46% | 92,22% |

O C25 argmax foi o melhor experimento completo em AUC, com 94,05%, indicando forte capacidade de ranking probabilístico. Já o C50_full calibrado foi o melhor experimento completo em acurácia, balanced accuracy e F1 macro, com 88,02%, 86,73% e 86,46%, respectivamente. Esse resultado foi selecionado como principal resultado downstream do trabalho.

A comparação entre C100, C25 e C50_full é central para a interpretação do método. O uso integral das sintéticas não produziu o melhor desempenho; ao contrário, C25 e C50_full superaram C100 nas métricas mais relevantes. Esse achado indica que as imagens geradas pelo LDM são úteis quando operam como augmentação controlada, mas podem diluir a distribuição real se incorporadas em excesso. Assim, a contribuição do sistema não está em substituir o domínio real por dados sintéticos, mas em introduzir variação adicional capaz de regularizar o classificador.

### 4.4 Matrizes de confusão e efeito da calibração

As matrizes de confusão do baseline A e do C50_full calibrado mostram a natureza do ganho obtido.

**Tabela 4 - Matrizes de confusão do baseline A e do C50_full calibrado**

| Modelo | Matriz de confusão |
|---|---|
| A | [[330, 159], [53, 961]] |
| C50_full calibrado | [[406, 83], [97, 917]] |

Considerando a ordem benigno/maligno, o C50_full calibrado reduziu substancialmente os falsos positivos benigno -> maligno, de 159 para 83, ao custo de aumento dos falsos negativos maligno -> benigno, de 53 para 97. Esse deslocamento explica o ganho expressivo em balanced accuracy: o modelo calibrado tornou a decisão mais equilibrada entre as duas classes, melhorando a recuperação da classe benigna, que era mais prejudicada no baseline.

Do ponto de vista prático, esse resultado deve ser interpretado com cautela. Em aplicações clínicas, falsos negativos malignos podem ter custo elevado, e a escolha do threshold dependeria do objetivo operacional. No contexto deste trabalho, o threshold calibrado é usado como instrumento experimental para avaliar a qualidade discriminativa do modelo sob um critério balanceado, não como recomendação clínica.

### 4.5 Testes estatísticos

A comparação estatística principal foi realizada entre o baseline binário A e o C50_full calibrado. O teste exato de McNemar por imagem indicou diferença significativa entre os classificadores, com p = 0,017169. Foram observados 69 casos em que A acertou e C50_full errou, contra 101 casos em que A errou e C50_full acertou, totalizando 170 pares discordantes.

**Tabela 5 - Testes estatísticos entre A e C50_full calibrado**

| Teste ou métrica | Resultado |
|---|---:|
| McNemar exato por imagem | p = 0,017169 |
| A correto / C50_full errado | 69 |
| A errado / C50_full correto | 101 |
| Bootstrap por paciente: accuracy | diff = 0,0219; IC95% [-0,0357; 0,0812] |
| Bootstrap por paciente: balanced accuracy | diff = 0,0618; IC95% [-0,0005; 0,1230] |
| Bootstrap por paciente: F1 macro | diff = 0,0411; IC95% [-0,0295; 0,1105] |
| Bootstrap por paciente: AUC | diff = 0,0325; IC95% [-0,0125; 0,1211] |

O bootstrap por paciente apresentou diferenças médias positivas para todas as métricas, mas os intervalos de confiança cruzaram ou praticamente tocaram zero. A balanced accuracy foi o caso mais próximo de significância, com IC95% [-0,0005; 0,1230]. Essa leitura é compatível com o tamanho reduzido do conjunto de teste, composto por 17 pacientes. Portanto, a conclusão estatística adequada é que há evidência favorável ao C50_full calibrado, significativa quando analisada por imagem, mas ainda conservadora no nível do paciente.

### 4.6 Qualidade gerativa das imagens sintéticas

A avaliação gerativa indicou que as imagens sintéticas não reproduziram perfeitamente a distribuição visual das imagens reais. O FID global amostrado foi 219,668. Nos subtipos avaliados, os valores de SSIM ficaram baixos, variando aproximadamente de 0,1095 a 0,1820, enquanto os valores de LPIPS foram altos, variando de 0,7789 a 0,8421. O subtipo ductal_carcinoma não recebeu imagens sintéticas porque já era a classe majoritária no conjunto de treinamento e serviu como referência para a equalização.

**Tabela 6 - Síntese das métricas gerativas**

| Métrica | Resultado |
|---|---:|
| FID global | 219,668 |
| SSIM por subtipo | 0,1095 a 0,1820 |
| LPIPS por subtipo | 0,7789 a 0,8421 |
| PSNR por subtipo | 7,6713 a 9,0130 |

Esses valores mostram que a qualidade gerativa, medida por métricas perceptuais e distribucionais, permaneceu limitada. Contudo, os resultados downstream indicam que essa limitação visual não impediu o uso das sintéticas como mecanismo de augmentação, reforçando a necessidade de avaliar imagens sintéticas por métricas, utilidade downstream e validação especializada, e não apenas por escores isolados [4,5]. A aparente tensão entre FID alto e melhora downstream é um dos achados relevantes do estudo: para esse protocolo, as imagens sintéticas não funcionaram como substitutas fiéis da distribuição real, mas como fonte controlada de variação capaz de melhorar a generalização binária.

### 4.7 Síntese dos resultados

Os resultados permitem três conclusões empíricas. Primeiro, o protocolo de oito subtipos não confirmou a hipótese original, sobretudo por limitações impostas pelo split por paciente e pelo número reduzido de pacientes em subtipos raros. Segundo, a formulação binária benigno versus maligno produziu um endpoint mais estável e defensável, no qual as imagens sintéticas por LDM melhoraram o desempenho downstream quando usadas em proporção controlada. Terceiro, a qualidade gerativa quantitativa foi limitada, com FID alto e métricas perceptuais desfavoráveis, de modo que o valor do método deve ser compreendido principalmente pelo impacto downstream, não pela fidelidade visual perfeita das amostras sintéticas.

## 5 Discussão

### 5.1 Refinamento da hipótese

A hipótese inicial previa que o sistema híbrido Encoder-Difusor superaria tanto o aumento clássico quanto o treinamento sem aumento, idealmente também no cenário de oito subtipos. Os resultados obtidos exigem uma formulação mais precisa, compatível com a literatura que descreve modelos de difusão em imagem médica como promissores, mas ainda dependentes de validação cuidadosa, desenho experimental e custos computacionais [2,6]. A hipótese foi parcialmente confirmada e refinada: o sistema proposto melhorou o desempenho downstream na classificação binária benigno versus maligno quando as imagens sintéticas foram usadas em proporção controlada, mas não validou plenamente a classificação em oito subtipos nem produziu imagens sintéticas de alta fidelidade segundo as métricas gerativas adotadas.

Esse refinamento não enfraquece necessariamente a contribuição do trabalho. Pelo contrário, torna a conclusão mais robusta. Em bases histopatológicas com poucos pacientes, a validade experimental depende de evitar vazamento entre treino e teste. Ao adotar split por paciente, o estudo expôs uma dificuldade que protocolos por imagem tendem a mascarar. Assim, o resultado negativo nos oito subtipos é parte importante da contribuição metodológica: ele mostra que altas granularidades diagnósticas podem ser estatisticamente frágeis quando há poucos pacientes por subtipo.

### 5.2 Valor das sintéticas como augmentação controlada

O achado mais importante do estudo é que as imagens sintéticas por LDM produziram ganho downstream quando incorporadas em proporção controlada. O C50_full calibrado superou o baseline A em acurácia, balanced accuracy e F1 macro, enquanto o C25 apresentou a maior AUC entre os experimentos completos. A comparação com C100 mostra que o benefício não é monotônico: adicionar mais dados sintéticos não levou automaticamente a melhor desempenho.

Essa observação é coerente com a interpretação das sintéticas como regularização ou augmentação, não como substituição do conjunto real, conforme estudos que descrevem dados sintéticos por LDM como força multiplicadora, mas não como substituto da coleta de dados reais diversos [3]. Em vez de tentar reconstruir a distribuição histopatológica completa, as imagens geradas parecem introduzir variações úteis para o classificador, especialmente em regiões de decisão afetadas pelo desbalanceamento. Quando a proporção sintética é excessiva, entretanto, o modelo pode ser exposto a padrões artificiais em volume suficiente para deslocar a aprendizagem em relação ao domínio real.

### 5.3 Relação entre qualidade gerativa e desempenho downstream

As métricas gerativas foram desfavoráveis: FID global de 219,668, SSIM baixo e LPIPS alto. Esses valores indicam distância relevante entre os domínios real e sintético. Caso o objetivo fosse substituir imagens reais por sintéticas ou produzir amostras indistinguíveis para uso diagnóstico, os resultados seriam insuficientes.

Entretanto, o objetivo downstream foi diferente: avaliar se as sintéticas poderiam melhorar o treinamento de classificadores. Nesse contexto, a melhora do C25 e do C50_full calibrado sugere que a utilidade de dados sintéticos não depende exclusivamente de fidelidade perceptual perfeita. Uma imagem sintética pode ser limitada visualmente e ainda assim atuar como perturbação estruturada útil ao treinamento. O ponto crítico é a proporção de uso, a seleção de classes e o critério de avaliação.

Essa distinção é essencial para a interpretação do trabalho. O sistema Encoder-Difusor não deve ser apresentado como gerador de imagens histopatológicas plenamente realistas, mas como uma estratégia experimental promissora de augmentação generativa controlada para classificação binária em cenário patient-wise.

### 5.4 Importância do split por paciente

A divisão por paciente foi uma decisão metodológica central. Ela reduziu o risco de data leakage e tornou as estimativas de desempenho mais conservadoras. No entanto, também evidenciou a fragilidade do BreakHis para classificação fina por subtipo: muitas classes possuem poucos pacientes, e um único indivíduo pode representar parcela expressiva de um subconjunto.

Essa constatação ajuda a explicar por que o cenário de oito subtipos apresentou macro F1 baixa mesmo quando a acurácia global parecia moderada. A acurácia foi influenciada pela classe majoritária, enquanto a macro F1 penalizou o fracasso nas classes raras. No contexto do TCC, essa diferença reforça a necessidade de reportar métricas balanceadas e de evitar conclusões baseadas apenas em acurácia.

### 5.5 Interpretação estatística

O teste de McNemar por imagem indicou diferença significativa entre A e C50_full calibrado. Ainda assim, a análise por paciente foi mais cautelosa: os efeitos foram positivos em todas as métricas, mas os intervalos de confiança cruzaram ou quase cruzaram zero. Esse resultado não invalida o ganho observado; ele delimita o grau de confiança que pode ser atribuído a ele.

Como o conjunto de teste contém apenas 17 pacientes, o bootstrap por paciente tem variância elevada. Portanto, a afirmação mais defensável é que o método apresentou evidência quantitativa favorável e promissora, com significância por imagem e tendência positiva por paciente, mas ainda requer validação em bases maiores ou com validação externa para sustentação clínica mais forte.

### 5.6 Limitações

As limitações do trabalho são relevantes e devem ser explicitadas. Primeiro, o VAE não foi fine-tunado no domínio histopatológico por restrição de hardware. O uso do `sd-vae-ft-mse` congelado tornou o estudo viável, mas pode ter limitado a fidelidade das reconstruções e das amostras geradas. Segundo, não foi implementado um baseline GAN diretamente comparável no mesmo protocolo. Assim, a comparação com GANs permanece fundamentada na literatura e na justificativa metodológica, não em uma comparação experimental local; esse limite deve ser preservado mesmo diante de estudos que mostram vantagem de modelos latentes de difusão sobre GANs em síntese médica multimodal [7].

Terceiro, não houve avaliação qualitativa por patologista. As métricas FID, SSIM, LPIPS e PSNR oferecem uma leitura quantitativa, mas não substituem julgamento especializado sobre plausibilidade histológica. Quarto, o protocolo de oito subtipos mostrou-se inviável ou instável sob split por paciente, especialmente por causa de subtipos com três a sete pacientes no total. Quinto, a qualidade gerativa medida foi limitada. Sexto, os testes estatísticos por paciente são conservadores e pouco potentes por causa dos 17 pacientes no conjunto de teste.

Essas limitações não anulam a contribuição do trabalho, mas definem seu escopo: trata-se de uma prova experimental promissora de augmentação generativa controlada, não de uma solução diagnóstica pronta para uso clínico.

### 5.7 Trabalhos futuros

Como próximos passos, recomenda-se: fine-tunar o VAE no domínio histopatológico em hardware com maior capacidade; avaliar o pipeline em bases externas; incluir avaliação qualitativa por patologistas; comparar diretamente com baselines GAN e com outros modelos de difusão; investigar estratégias de seleção ou filtragem das sintéticas antes do treinamento downstream; estudar limiares de decisão ajustados a custos clínicos, especialmente para reduzir falsos negativos malignos; e avaliar condicionamentos histopatológicos mais ricos, como anotações derivadas de imagem, já exploradas em LDMs para patologia [4,8].

## Conclusão

Este trabalho implementou e avaliou um sistema híbrido baseado em Encoder e Difusor para aumento de dados em imagens histopatológicas de câncer de mama do BreakHis. O pipeline incluiu organização rigorosa da base, split estratificado por paciente, uso de VAE pré-treinado para compressão latente, treinamento de um LDM condicionado por subtipo histológico, geração de imagens sintéticas, avaliação gerativa quantitativa e avaliação downstream com classificadores supervisionados.

A hipótese inicial, formulada como superioridade ampla do sistema proposto sobre o baseline e o aumento clássico, precisou ser refinada. No cenário de oito subtipos, os resultados não confirmaram a hipótese: o baixo número de pacientes em classes raras tornou a tarefa instável sob split patient-wise, levando a baixo F1 macro e a forte influência da classe majoritária. Esse resultado, embora negativo, foi metodologicamente relevante por evidenciar a fragilidade de avaliações por subtipo quando a divisão por paciente é preservada.

Na formulação binária benigno versus maligno, os resultados foram mais favoráveis. O melhor experimento completo, C50_full calibrado, alcançou 88,02% de acurácia, 86,73% de balanced accuracy, 86,46% de F1 macro e 92,22% de AUC. O C25 obteve a maior AUC entre os experimentos completos, com 94,05%. A comparação estatística entre o baseline A e o C50_full calibrado apresentou significância pelo teste de McNemar por imagem, com p = 0,017169, e efeitos positivos no bootstrap por paciente, embora com intervalos de confiança ainda conservadores devido ao número reduzido de pacientes no teste.

A avaliação gerativa mostrou limitações importantes. O FID global de 219,668, combinado a SSIM baixo e LPIPS alto, indica que as imagens sintéticas não substituem adequadamente a distribuição visual real. Ainda assim, a melhora downstream observada nos cenários C25 e C50_full calibrado sugere que as sintéticas podem ser úteis como augmentação controlada. Portanto, o valor do método não está em produzir imagens perfeitas, mas em introduzir variação sintética suficiente para melhorar a generalização do classificador em um protocolo binário rigoroso.

Conclui-se que a hipótese foi parcialmente confirmada e refinada. O sistema Encoder-Difusor implementado mostrou-se promissor para aumento de dados em classificação binária de histopatologia mamária, desde que as imagens sintéticas sejam usadas em proporção controlada e avaliadas por seu impacto downstream. O trabalho também evidencia que protocolos de avaliação em imagens médicas devem priorizar divisão por paciente, métricas balanceadas e análise estatística compatível com a unidade real de generalização. Como contribuição final, o estudo oferece um pipeline reprodutível e uma leitura crítica dos limites e possibilidades de modelos de difusão latente para augmentação de dados em histopatologia.
## Referências verificadas no Consensus

As referências abaixo foram recuperadas no Consensus e podem ser integradas à seção bibliográfica final em formato ABNT. Uso sugerido: apoiar o split patient-wise, a justificativa de aumento generativo por difusão, a interpretação de dados sintéticos como augmentação controlada, a cautela com FID/LPIPS e a ausência de baseline GAN próprio.

[1] [A patient-aware benchmarking of CNN and transformer architectures for breast cancer histopathology classification](https://consensus.app/papers/a-patientaware-benchmarking-of-cnn-and-transformer-priyanka-narendra/e86c2e160df557e8854909d5a9ba54ca/?utm_source=chatgpt). Veeram Priyanka, Modigari Narendra, Tharasi Dilleswar Rao. 2026. Frontiers in Digital Health, v. 8. Citações: 0. Uso no TCC: justificar split patient-wise e risco de data leakage em BreakHis.

[2] [Deep Learning Approaches for Data Augmentation in Medical Imaging: A Review](https://consensus.app/papers/deep-learning-approaches-for-data-augmentation-in-medical-kebaili-lapuyade-lahorgue/4319b8d4f33556658086ae1a2352fa6a/?utm_source=chatgpt). Aghiles Kebaili, J. Lapuyade-Lahorgue, S. Ruan. 2023. Journal of Imaging, v. 9. Citações: 272. Uso no TCC: contextualizar VAEs, GANs e modelos de difusão como aumento generativo em imagens médicas.

[3] [Augmenting medical image classifiers with synthetic data from latent diffusion models](https://consensus.app/papers/augmenting-medical-image-classifiers-with-synthetic-data-sagers-diao/18629e8691cc5631a9fccfd2809d3b63/?utm_source=chatgpt). Luke Sagers, James A. Diao, Luke Melas-Kyriazi, Matthew Groh, P. Rajpurkar, A. Adamson, V. Rotemberg, Roxana Daneshjou, A. Manrai. 2023. ArXiv, abs/2308.12453. Citações: 24. Uso no TCC: apoiar a tese de dados sintéticos como augmentação controlada/força multiplicadora, sem substituir dados reais.

[4] [Generating and evaluating synthetic data in digital pathology through diffusion models](https://consensus.app/papers/generating-and-evaluating-synthetic-data-in-digital-pozzi-noei/2b3ab42072185c34bff6c6b5f572e891/?utm_source=chatgpt). M. Pozzi, S. Noei, E. Robbi, L. Cima, M. Moroni, E. Munari, E. Torresani, G. Jurman. 2024. Scientific Reports, v. 14. Citações: 30. Uso no TCC: defender avaliação multifacetada em patologia digital, combinando métricas, utilidade downstream, explicabilidade e avaliação por patologistas.

[5] [Evaluating Synthetic Medical Images Using Artificial Intelligence with the GAN Algorithm](https://consensus.app/papers/evaluating-synthetic-medical-images-using-artificial-abdusalomov-nasimov/e03f0bbca60e5deeae68657b60333246/?utm_source=chatgpt). A. Abdusalomov, R. Nasimov, N. Nasimova, Bahodir Muminov, T. Whangbo. 2023. Sensors (Basel, Switzerland), v. 23. Citações: 52. Uso no TCC: sustentar a cautela sobre métricas isoladas para adequação médica de imagens sintéticas.

[6] [Diffusion models in medical imaging: A comprehensive survey](https://consensus.app/papers/diffusion-models-in-medical-imaging-a-comprehensive-kazerouni-aghdam/eb1b270b9c3f5b8a902ab6ef44145064/?utm_source=chatgpt). A. Kazerouni, E. K. Aghdam, Moein Heidari, Reza Azad, I. Hacihaliloglu, D. Merhof. 2022. Medical Image Analysis, v. 88, artigo 102846. Citações: 721. Uso no TCC: revisão ampla sobre modelos de difusão em imagem médica.

[7] [A multimodal comparison of latent denoising diffusion probabilistic models and generative adversarial networks for medical image synthesis](https://consensus.app/papers/a-multimodal-comparison-of-latent-denoising-diffusion-mller-franzes-niehues/0ad159d141ed5945a96359303504a980/?utm_source=chatgpt). Gustav Müller-Franzes, J. Niehues, Firas Khader, Soroosh Tayebi Arasteh, Christoph Haarburger, C. Kuhl, Tian Wang, T. Han, S. Nebelung, Jakob Nikolas Kather, D. Truhn. 2022. Scientific Reports, v. 13. Citações: 207. Uso no TCC: apoiar a escolha por difusão frente a GANs, sem apresentar como baseline experimental próprio.

[8] [Latent Diffusion Models with Image-Derived Annotations for Enhanced AI-Assisted Cancer Diagnosis in Histopathology](https://consensus.app/papers/latent-diffusion-models-with-imagederived-annotations-osrio-jimnez-prez/e442790a1772529c9fab8364561f622a/?utm_source=chatgpt). Pedro Osório, Guillermo Jiménez-Pérez, Javier Montalt-Tordera, Jens Hooge, Guillem Duran Ballester, Shivam Singh, Moritz Radbruch, Ute Bach, S. Schroeder, K. Siudak, Julia Vienenkoetter, Bettina Lawrenz, Sadegh Mohammadi. 2023. Diagnostics, v. 14. Citações: 17. Uso no TCC: contextualizar LDMs em histopatologia e geração com anotações derivadas de imagem.

### Pontos de integração bibliográfica no texto

- Metodologia, split por paciente: citar [1] ao justificar que splits por imagem podem produzir data leakage e estimativas otimistas no BreakHis.
- Metodologia, aumento generativo: citar [2] e [6] ao apresentar VAEs, GANs e difusão como famílias generativas aplicadas a imagens médicas.
- Discussão, uso controlado de sintéticas: citar [3] ao afirmar que sintéticas podem atuar como multiplicador de dados, mas não substituem a coleta de dados reais diversos.
- Resultados/Discussão, FID alto: citar [4] e [5] ao defender que FID/LPIPS/SSIM são úteis, mas insuficientes sozinhos para determinar adequação clínica ou histopatológica.
- Discussão, GANs versus difusão: citar [7] como suporte bibliográfico para a escolha conceitual por difusão, deixando explícito que este TCC não implementou baseline GAN local.
- Trabalhos futuros: citar [4] e [8] ao recomendar avaliação por patologistas e condicionamento histopatológico mais rico.


