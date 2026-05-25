# Answer-Preserving Qualitative Examples

Rows are accessible-steering interventions with unchanged selected top-1 token and full-vocabulary top-10 Jaccard at least 0.90.

## decrease | bert-base-uncased | prompt 9

- Prompt: `The meeting happened near the [MASK] in the school.`
- Preserved answer token: `library`
- Entropy: `0.9675 -> 0.8839` (`Delta=-0.0836`)
- Varentropy: `3.2794 -> 3.1486` (`Delta=-0.1308`)
- rho: `0.9609`; top-10 Jaccard: `1.000`; candidate KL: `0.001151`

## decrease | bert-base-uncased | prompt 14

- Prompt: `The capital of Austria is [MASK].`
- Preserved answer token: `vienna`
- Entropy: `2.1028 -> 2.0231` (`Delta=-0.0797`)
- Varentropy: `2.9093 -> 3.0048` (`Delta=0.0956`)
- rho: `0.8678`; top-10 Jaccard: `1.000`; candidate KL: `0.00124`

## decrease | google/bert_uncased_L-2_H-128_A-2 | prompt 20

- Prompt: `The language of Germany is [MASK].`
- Preserved answer token: `germany`
- Entropy: `1.5592 -> 1.4804` (`Delta=-0.0788`)
- Varentropy: `3.2307 -> 3.2246` (`Delta=-0.0061`)
- rho: `0.8020`; top-10 Jaccard: `1.000`; candidate KL: `0.001214`

## decrease | bert-base-uncased | prompt 3

- Prompt: `The coach talked to the [MASK].`
- Preserved answer token: `coach`
- Entropy: `2.2679 -> 2.1914` (`Delta=-0.0765`)
- Varentropy: `2.6923 -> 2.8135` (`Delta=0.1213`)
- rho: `0.8726`; top-10 Jaccard: `1.000`; candidate KL: `0.001216`

## decrease | bert-base-uncased | prompt 3

- Prompt: `The coach talked to the [MASK].`
- Preserved answer token: `coach`
- Entropy: `2.2679 -> 2.1918` (`Delta=-0.0761`)
- Varentropy: `2.6923 -> 2.8045` (`Delta=0.1122`)
- rho: `0.8560`; top-10 Jaccard: `1.000`; candidate KL: `0.001231`

## decrease | bert-base-uncased | prompt 22

- Prompt: `The currency of Spain is the [MASK].`
- Preserved answer token: `euro`
- Entropy: `1.7421 -> 1.6672` (`Delta=-0.0749`)
- Varentropy: `3.1923 -> 3.2225` (`Delta=0.0302`)
- rho: `0.8000`; top-10 Jaccard: `1.000`; candidate KL: `0.001106`

## decrease | bert-base-uncased | prompt 10

- Prompt: `The opposite of early is [MASK].`
- Preserved answer token: `late`
- Entropy: `2.0254 -> 1.9514` (`Delta=-0.0740`)
- Varentropy: `2.6432 -> 2.6629` (`Delta=0.0197`)
- rho: `0.8685`; top-10 Jaccard: `1.000`; candidate KL: `0.001198`

## decrease | distilbert-base-uncased | prompt 3

- Prompt: `The coach talked to the [MASK].`
- Preserved answer token: `referee`
- Entropy: `2.5250 -> 2.4510` (`Delta=-0.0739`)
- Varentropy: `2.3780 -> 2.5241` (`Delta=0.1461`)
- rho: `0.8939`; top-10 Jaccard: `1.000`; candidate KL: `0.001243`

## decrease | bert-base-uncased | prompt 11

- Prompt: `Ten divided by two equals [MASK].`
- Preserved answer token: `ten`
- Entropy: `2.4088 -> 2.3355` (`Delta=-0.0733`)
- Varentropy: `2.4648 -> 2.5905` (`Delta=0.1257`)
- rho: `0.8644`; top-10 Jaccard: `1.000`; candidate KL: `0.001229`

## decrease | bert-base-uncased | prompt 3

- Prompt: `The coach talked to the [MASK].`
- Preserved answer token: `coach`
- Entropy: `2.2679 -> 2.1950` (`Delta=-0.0729`)
- Varentropy: `2.6923 -> 2.8120` (`Delta=0.1197`)
- rho: `0.7886`; top-10 Jaccard: `1.000`; candidate KL: `0.001226`

## decrease | distilbert-base-uncased | prompt 22

- Prompt: `The currency of Spain is the [MASK].`
- Preserved answer token: `euro`
- Entropy: `0.6931 -> 0.6206` (`Delta=-0.0725`)
- Varentropy: `2.6271 -> 2.4625` (`Delta=-0.1646`)
- rho: `0.9020`; top-10 Jaccard: `1.000`; candidate KL: `0.001173`

## decrease | distilbert-base-uncased | prompt 22

- Prompt: `The currency of Spain is the [MASK].`
- Preserved answer token: `euro`
- Entropy: `0.6931 -> 0.6211` (`Delta=-0.0720`)
- Varentropy: `2.6271 -> 2.4599` (`Delta=-0.1672`)
- rho: `0.9470`; top-10 Jaccard: `1.000`; candidate KL: `0.001101`

## increase | bert-base-uncased | prompt 9

- Prompt: `The meeting happened near the [MASK] in the school.`
- Preserved answer token: `library`
- Entropy: `0.9675 -> 1.0590` (`Delta=0.0915`)
- Varentropy: `3.2794 -> 3.4032` (`Delta=0.1238`)
- rho: `0.9433`; top-10 Jaccard: `1.000`; candidate KL: `0.001307`

## increase | bert-base-uncased | prompt 9

- Prompt: `The meeting happened near the [MASK] in the school.`
- Preserved answer token: `library`
- Entropy: `0.9675 -> 1.0542` (`Delta=0.0867`)
- Varentropy: `3.2794 -> 3.3821` (`Delta=0.1027`)
- rho: `0.8484`; top-10 Jaccard: `1.000`; candidate KL: `0.001307`

## increase | bert-base-uncased | prompt 22

- Prompt: `The currency of Spain is the [MASK].`
- Preserved answer token: `euro`
- Entropy: `1.7421 -> 1.8275` (`Delta=0.0853`)
- Varentropy: `3.1923 -> 3.1410` (`Delta=-0.0513`)
- rho: `0.8772`; top-10 Jaccard: `1.000`; candidate KL: `0.001301`

## increase | bert-base-uncased | prompt 8

- Prompt: `A person who drives a bus is a [MASK].`
- Preserved answer token: `driver`
- Entropy: `1.8689 -> 1.9500` (`Delta=0.0811`)
- Varentropy: `3.0522 -> 2.9875` (`Delta=-0.0647`)
- rho: `0.8259`; top-10 Jaccard: `1.000`; candidate KL: `0.001309`
