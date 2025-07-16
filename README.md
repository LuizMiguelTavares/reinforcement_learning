# Reinforcement Learning – Workspace ROS

Este repositório contém os **pacotes ROS** e **scripts** utilizados nos nossos experimentos de Aprendizado por Reforço com o robô **LIMO** da AgileX.

```
reinforcement_learning/          # raiz do workspace
├── limo_sim/                    # simulador oficial da AgileX (pacote ROS)
├── r_learning/                  # onde ficam todos os nossos códigos
│   ├── launch/
│   ├── maps/
│   ├── scripts/
│   └── src/
└── ReinforcementLearning/       # diretório extra; contém notebooks e protótipos (NÃO é um pacote ROS)
```

> **Observação:** sempre trabalhe no workspace já carregado com `catkin_ws/devel/setup.bash`.

---

## 1. Criação de mapas

O **launch file** `create_map.launch` abre uma interface gráfica para que você desenhe obstáculos e gere um mapa de ocupação:

```bash
roslaunch r_learning create_map.launch
```

* Ao fechar a janela, o mapa é salvo automaticamente em `$(find r_learning)/maps/`.
* Para alterar as dimensões ou o nome do arquivo, use argumentos:

```bash
roslaunch r_learning create_map.launch \
  grid_width:=6  grid_height:=5  cell_size_cm:=5  file_name:=meu_mapa
```

| Argumento      | Descrição                                      | Padrão         |
| -------------- | ---------------------------------------------- | -------------- |
| `grid_width`   | Largura em células                             | 9              |
| `grid_height`  | Altura em células                              | 7              |
| `cell_size_cm` | Tamanho da célula em **cm**                    | 50              |
| `file_name`    | Nome do arquivo **.yaml** no diretório `maps/` | `limo_lab_map` |

> **Cuidado:** se você reutilizar `file_name` sem alterar, o mapa anterior será sobrescrito.

---

## 2. Simulação do LIMO + mapa

Execute o launch `limo_and_map.launch` para:

1. Iniciar o Gazebo com o modelo do LIMO
2. Carregar o mapa escolhido
3. Abrir o RViz com as configurações corretas
4. Iniciar o script `publish_gazebo_positions.py`, que publica a pose “real” do robô em
   `/vrpn_client_node/P1/pose` (simulando um OptiTrack)

```bash
roslaunch r_learning limo_and_map.launch
```

### Argumentos úteis

| Argumento  | Descrição                                                  | Padrão                                      |
| ---------- | ---------------------------------------------------------- | ------------------------------------------- |
| `gui`      | Exibir ou não a interface gráfica do Gazebo (`true/false`) | `false`                                     |
| `use_rviz` | Abrir o RViz (`true/false`)                                | `true`                                      |
| `map`      | Caminho para o arquivo `.yaml` do mapa                     | `$(find r_learning)/maps/limo_lab_map.yaml` |
| `x`, `y`   | Posição inicial do robô                                    | `0.25`                                      |
| `yaw`      | Orientação inicial (rad)                                   | `0.0`                                       |

Exemplo alterando mapa e spawn:

```bash
roslaunch r_learning limo_and_map.launch \
  map:=$(rospack find r_learning)/maps/meu_mapa.yaml \
  x:=1.0  y:=0.5  yaw:=1.57
```

---

## 3. Planejamento de caminho (RL)

Com a simulação em execução, inicie o módulo de Aprendizado por Reforço:

```bash
roslaunch r_learning rl_miguel.launch
```

Isso executa o script `scripts/rl_miguel.py`, responsável por gerar o caminho até o objetivo.

---

## 4. Controle do robô

Para seguir o caminho planejado, execute:

```bash
roslaunch r_learning differential_control.launch
```

* Este launch roda o nó de controle (`src/differential_control.cpp`) **e** um botão de emergência.
* A janela do botão permite parar o LIMO imediatamente, se necessário.

---

## 5. Reiniciar a simulação rapidamente

Não é preciso reiniciar o Gazebo entre testes. Após o robô alcançar o objetivo, basta resetar o mundo:

```bash
rosservice call /gazebo/reset_world "{}"
```

O LIMO volta à posição inicial e você pode iniciar um novo experimento.

---

## 6. Boas práticas

1. **Um mapa para vários testes:** carregue o mapa uma vez e reutilize-o.
2. **Versão de arquivo:** se precisar gerar vários mapas, use nomes diferentes.
3. **Economia de recursos:** mantenha `gui:=false` no Gazebo se usar apenas o RViz.
