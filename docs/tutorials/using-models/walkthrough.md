## AgentTorch API

- Execute simulations
- Customize agents
- Customize populations
- Engage with results


```python
import pandas as pd
import os
```

#### Simulation Input

`Simulation Input` : Input to our simulation is a dataset consisting of population generated based on the census data of a specific region. We have currently generated population for New York City and New Zealand.


Lets see how the population data looks like for New Zealand


```python
NZ_SYNTHETIC_POPULATION_PATH = os.path.join(os.getcwd(), 'simulator_data/census_populations/NZ/synthetic_populations/population_data.pkl')

nz_population = pd.read_pickle(NZ_SYNTHETIC_POPULATION_PATH)
nz_population.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>age</th>
      <th>gender</th>
      <th>ethnicity</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>204200</td>
      <td>U19</td>
      <td>female</td>
      <td>Maori</td>
      <td>Bay of Plenty</td>
    </tr>
    <tr>
      <th>1</th>
      <td>204200</td>
      <td>U19</td>
      <td>female</td>
      <td>Maori</td>
      <td>Bay of Plenty</td>
    </tr>
    <tr>
      <th>2</th>
      <td>204200</td>
      <td>U19</td>
      <td>male</td>
      <td>European</td>
      <td>Bay of Plenty</td>
    </tr>
    <tr>
      <th>3</th>
      <td>204200</td>
      <td>U19</td>
      <td>female</td>
      <td>Maori</td>
      <td>Bay of Plenty</td>
    </tr>
    <tr>
      <th>4</th>
      <td>204200</td>
      <td>U19</td>
      <td>female</td>
      <td>Maori</td>
      <td>Bay of Plenty</td>
    </tr>
  </tbody>
</table>
</div>



The above data shows the population attributes of 5 agents, like 'age', 'gender', 'region where the agent resides', 'ethnicity' and 'area code of residence'. We assign these attributes to each of our simulated agents, to mimic real world population dynamic.

We also provide APIs to generate your own synthetic population, though we will not get into the details here.

## Run Simulation

`Dataloader` is the common interface to load, setup or modify the simulation parameters such as Number of Runs, Population Size, Region for which Population data is to be used. It also does the necessary pipelining to setup the flow of simulation.

It compiles a config file which is used by the Executor to run the simulation.


```python
from AgentTorch.dataloader import DataLoader

# 'populations' is the package where generated population for each region is saved.
# We are picking New Zealand's (NZ) population for this run.
from populations import NZ

# 'models' is the package where the simulation logic for different use cases are defined.
# We are picking the macro_economics model for this simulation. It models the population's economic behavior during covid, by simulating a financial market.
from models import macro_economics
from macro_economics.config import NUM_AGENTS

# Initialize the data loader with the preferred model and the region's population.
data_loader = DataLoader(model = macro_economics, region = NZ, population_size = NUM_AGENTS)

```

    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml
    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml
    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml


Example 1: Use `Executor` API to run a simulation


```python
# Executor is the common interface to run simulation
# It takes the model and data loader as input and executes the simulation.
from AgentTorch.execute import Executor

agents_sim = Executor(model = macro_economics,data_loader = data_loader)
agents_sim.execute()

```

    
    running episode 0...
    Substep Action: Earning decision
    LLM benchmark expts
    Some Example Prompts:  ['You are female of age U19, living in the Bay of Plenty region. It is September 2020, number of covid cases is 874.0. The price of Essential Goods is 12.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Otago region. It is September 2020, number of covid cases is 874.0. The price of Essential Goods is 12.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Tasman region. It is September 2020, number of covid cases is 874.0. The price of Essential Goods is 12.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ']
    Substep: Agent Earning!
    Substep: Agent Consumption
    Executing Substep: Labor Market
    Executing Substep: Financial Market
    Substep Action: Earning decision
    LLM benchmark expts
    Some Example Prompts:  ['You are female of age U19, living in the Bay of Plenty region. It is October 2020, number of covid cases is 701.0. The price of Essential Goods is 12.012574195861816. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Otago region. It is October 2020, number of covid cases is 701.0. The price of Essential Goods is 12.012574195861816. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Tasman region. It is October 2020, number of covid cases is 701.0. The price of Essential Goods is 12.012574195861816. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ']
    Substep: Agent Earning!
    Substep: Agent Consumption
    Executing Substep: Labor Market
    Executing Substep: Financial Market
    Simulation Ended.


Example 2: Use `DataLoader` API to customize the agent - Make LLM an Agent

Let's modify the prompt used by our LLM Agent, and re-run the simulation

Original Prompt is : "You are {gender} of age {age}, living in the {region} region. It is {month} {year}, number of covid cases is {covid_cases}. The price of Essential Goods is {price_of_goods}. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? "

We will modify this to remove the information about `price of essential goods`. This might, help us understand the population/market behaviour when info on inflation is not present.


```python
MODIFIED_EARNING_ACTION_PROMPT = "You are {gender} of age {age}, living in the {region} region. It is {month} {year}, number of covid cases is {covid_cases}. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? "
data_loader.set_config_attribute('EARNING_ACTION_PROMPT', MODIFIED_EARNING_ACTION_PROMPT)

agent_sim_modified = Executor(model = macro_economics,data_loader = data_loader)
# With this set, we can now run the simulation again using the executor.
agent_sim_modified.execute()
```

    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml
    resolvers already registered..
    
    running episode 0...
    Substep Action: Earning decision
    LLM benchmark expts
    Some Example Prompts:  ['You are female of age U19, living in the Bay of Plenty region. It is September 2020, number of covid cases is 874.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Otago region. It is September 2020, number of covid cases is 874.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Tasman region. It is September 2020, number of covid cases is 874.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ']
    Substep: Agent Earning!
    Substep: Agent Consumption
    Executing Substep: Labor Market
    Executing Substep: Financial Market
    Substep Action: Earning decision
    LLM benchmark expts
    Some Example Prompts:  ['You are female of age U19, living in the Bay of Plenty region. It is October 2020, number of covid cases is 701.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Otago region. It is October 2020, number of covid cases is 701.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are female of age U19, living in the Tasman region. It is October 2020, number of covid cases is 701.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ']
    Substep: Agent Earning!
    Substep: Agent Consumption
    Executing Substep: Labor Market
    Executing Substep: Financial Market
    Simulation Ended.


## Visualize Results

Simulations produce a state trace
 
 More info on state trace ....


```python
from macro_economics.config import STATE_TRACE_PATH
from AgentTorch.LLM.llm_qa import load_state_trace

state_trace = load_state_trace(STATE_TRACE_PATH,NZ_NUM_AGENTS)
state_trace[0].columns
```




    Index(['ID', 'age', 'area', 'assets', 'consumption_propensity', 'ethnicity',
           'gender', 'household', 'monthly_income', 'post_tax_income', 'region',
           'will_work', 'month', 'year'],
          dtype='object')



Example 3: Use `Visualizer` API to engage the simulation trace for retrospective and prospective analysis 


```python
# Visualizer is the helper class to visualize the output variables of the simulation
# As we would expect, it takes simulation trace as input
from AgentTorch.visualize import Visualizer
visualizer = Visualizer(state_trace)
```


```python
# We can plot a few variables of interest like Consumption Propensity and Will to Work.
visualizer.plot('consumption_propensity')
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_6f7442dad93dd4d369ae52e936a40710 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_6f7442dad93dd4d369ae52e936a40710&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_6f7442dad93dd4d369ae52e936a40710 = L.map(
                &quot;map_6f7442dad93dd4d369ae52e936a40710&quot;,
                {
                    center: [-40.0, 172.0],
                    crs: L.CRS.EPSG3857,
                    zoom: 5,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_1d1fddaaa061236be84ce3d51a358ae6 = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_1d1fddaaa061236be84ce3d51a358ae6.addTo(map_6f7442dad93dd4d369ae52e936a40710);


            var circle_marker_6bd3e7f2a8c03895f7cb50c3591b6bd7 = L.circleMarker(
                [-34.505453, 172.77555],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;orange&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;orange&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_6f7442dad93dd4d369ae52e936a40710);


            var circle_marker_c2c11d8503d7c0471c7b5b1b5ff9ea36 = L.circleMarker(
                [-34.916277, 173.137443],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;red&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;red&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_6f7442dad93dd4d369ae52e936a40710);


            var circle_marker_1e5f3b6283285e0dd2bc1dd947ce2cd9 = L.circleMarker(
                [-35.218501, 174.158249],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;yellow&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;yellow&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_6f7442dad93dd4d369ae52e936a40710);


            var circle_marker_5bfd1fd87a0040f91c72474798e6c960 = L.circleMarker(
                [-34.995278, 173.378738],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;darkred&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;darkred&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_6f7442dad93dd4d369ae52e936a40710);


            var circle_marker_48eae7c7d4a0b8801799060b2212c546 = L.circleMarker(
                [-35.123147, 173.218604],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;green&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;green&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_6f7442dad93dd4d369ae52e936a40710);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
# Next lets visualize the Aggregate Willingness to work in each region
visualizer.plot('will_to_work')
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_56b0f3742038f3d699b1c7ba425bb756 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_56b0f3742038f3d699b1c7ba425bb756&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_56b0f3742038f3d699b1c7ba425bb756 = L.map(
                &quot;map_56b0f3742038f3d699b1c7ba425bb756&quot;,
                {
                    center: [-40.0, 172.0],
                    crs: L.CRS.EPSG3857,
                    zoom: 5,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_01457a4ab25c0427185b19426e8cbc06 = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_01457a4ab25c0427185b19426e8cbc06.addTo(map_56b0f3742038f3d699b1c7ba425bb756);


            var circle_marker_7b101f362620c1cbc169e27cf361564d = L.circleMarker(
                [-34.505453, 172.77555],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;yellow&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;yellow&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_56b0f3742038f3d699b1c7ba425bb756);


            var circle_marker_97f7861d0d4123f39a7e14f08b4772b1 = L.circleMarker(
                [-34.916277, 173.137443],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;red&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;red&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_56b0f3742038f3d699b1c7ba425bb756);


            var circle_marker_172ace1ef8100ec6c2d761f444bfbfd5 = L.circleMarker(
                [-35.218501, 174.158249],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;green&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;green&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_56b0f3742038f3d699b1c7ba425bb756);


            var circle_marker_279a095d05a66024861bd8483efb50b0 = L.circleMarker(
                [-34.995278, 173.378738],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;red&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;red&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_56b0f3742038f3d699b1c7ba425bb756);


            var circle_marker_ef047d26f4aad78ec3e8187218a3f1b4 = L.circleMarker(
                [-35.123147, 173.218604],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;yellow&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;yellow&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 20, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_56b0f3742038f3d699b1c7ba425bb756);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



## Talk with Simulation

`Q/A` : AgentTorch gives you the ability to perform Q/A with the simulation, through its integration of simulation trace and conversation history with th Analysis Agent (LLM Agent).

##### Configuration


```python
from macro_economics.config import OPENAI_API_KEY
conversations_memory_directory = ROOT_DIR + 'populations/NZ/simulation_memory_output/2020/9'
state_trace_path = ROOT_DIR + 'populations/NZ/state_data_dict.pkl'
```

##### Analyze

`DocumentRetriever` : It sets up a vector db containing all the converations intiated with the LLM Agent, during the runtime of the simulation.


```python
from AgentTorch.LLM.llm_qa import DocumentRetriever
conversation_memory_retriever = DocumentRetriever(directory=conversation_memory_directory)
```


    Batches:   0%|          | 0/16 [00:00<?, ?it/s]


`SimulationAnalysisAgent` abstracts away the logic for analysis module, so you can do what's more fun, that is analyzing the simulation behaviour. 


```python
from AgentTorch.LLM.llm_qa import SimulationAnalysisAgent
analyser = SimulationAnalysisAgent(openai_api_key=OPENAI_API_KEY, document_retriever=memory_retriever, temperature=0, state_trace_path=state_trace_path)
```


```python
analyser.query("Which age group has lowest median income, how much is it?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `run_analysis_on_simulation_state` with `{'query': 'Which age group has the lowest median income and how much is it?'}`
    
    
    [0m[36;1m[1;3mThe age group with the lowest median income is 20t29 with a median income of $168.00.[0m[32;1m[1;3mThe age group with the lowest median income is 20-29 years old, with a median income of $168.00.[0m
    
    [1m> Finished chain.[0m





    {'input': 'Which age group has lowest median income, how much is it?',
     'output': 'The age group with the lowest median income is 20-29 years old, with a median income of $168.00.'}




```python
analyser.query("Which age group has highest consumption")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `run_analysis_on_simulation_state` with `{'query': 'Which age group has the highest consumption?'}`
    
    
    [0m[36;1m[1;3mThe age group with the highest consumption is 65A.[0m[32;1m[1;3mThe age group with the highest consumption is 65 and above.[0m
    
    [1m> Finished chain.[0m





    {'input': 'Which age group has highest consumption',
     'output': 'The age group with the highest consumption is 65 and above.'}




```python
analyser.query("How is inflation affecting consumption behaviour?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `simulation_memory_retriever` with `{'query': 'inflation and consumption behaviour'}`
    
    
    [0m


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    [33;1m[1;3mReasoning: Let's think step by step in order to[32m produce the answer. We observe a decrease in the number of COVID cases from 874.0 to 701.0, indicating a potential improvement in the pandemic situation in the Wellington region. This improvement might lead to a higher willingness to work due to perceived lower health risks and possibly more job opportunities as the economy might start to recover. The slight increase in the price of Essential Goods from 12.0 to 12.012574195861816 suggests a minimal inflation in the cost of living, which might not significantly impact the overall expenditure strategy but indicates economic activity. Given the age group (30t39), there's likely a strong incentive to maintain or increase income levels to support any current or future financial goals, such as savings, investments,[0m
    
    Reasoning: Let's think step by step in order to[32m produce the answer. Given the decrease in the number of COVID cases from 874.0 to 701.0, there is a slight improvement in the pandemic situation in the Waikato region. This improvement might lead to a more stable economic environment and potentially more job security. As a male in the age group of 40 to 49, you are likely to have significant financial responsibilities, possibly including a family to support, a mortgage, and future savings to consider. The slight increase in the price of Essential Goods from 12.0 to 12.012574195861816 indicates a marginal inflation but doesn't significantly impact the overall cost of living. Considering these factors:
    
    1. **Willingness to Work**: With the decrease in[0m
    
    Reasoning: Let's think step by step in order to produce the answer. We observe a decrease in the number of COVID cases from 874.0 to 701.0, indicating a potential improvement in the pandemic situation in the Wellington region. This improvement might lead to a higher willingness to work due to perceived lower health risks and possibly more job opportunities as the economy might start to recover. The slight increase in the price of Essential Goods from 12.0 to 12.012574195861816 suggests a minimal inflation in the cost of living, which might not significantly impact the overall expenditure strategy but indicates economic activity. Given the age group (30t39), there's likely a strong incentive to maintain or increase income levels to support any current or future financial goals, such as savings, investments,
    
    Answer:[32m [0.75, 0.6][0m
    
    Reasoning: Let's think step by step in order to produce the answer. Given the decrease in the number of COVID cases from 874.0 to 701.0, there is a slight improvement in the pandemic situation in the Waikato region. This improvement might lead to a more stable economic environment and potentially more job security. As a male in the age group of 40 to 49, you are likely to have significant financial responsibilities, possibly including a family to support, a mortgage, and future savings to consider. The slight increase in the price of Essential Goods from 12.0 to 12.012574195861816 indicates a marginal inflation but doesn't significantly impact the overall cost of living. Considering these factors: 1. **Willingness to Work**: With the decrease in
    
    Answer:[32m [0.7, 0.7][0m
    
    Reasoning: Let's think step by step in order to produce the answer. We need to consider several factors: 1. **Age and Health Risk**: At age 65, you are in a higher risk category for COVID-19. This increases the risk associated with working, especially if your job cannot be done remotely. 2. **COVID-19 Cases**: The number of cases in the Otago region has decreased from 874.0 to 701.0. This indicates a positive trend, but the risk of infection is still present. 3. **Price of Essential Goods**: The price has slightly increased from 12.0 to 12.012574195861816. This indicates inflation but not at a significant rate that would drastically change consumption habits. 4. **Economic Trends and
    
    Answer:[32m [0.25, 0.65][0m
    
    
    
    
    
    
    
    You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assests you are willing to spend to meet your consumption demands, based on the current situation of NYC.
    
    ---[0m[32;1m[1;3m
    Invoking: `run_analysis_on_simulation_state` with `{'query': 'impact of inflation on consumption behaviour'}`
    
    
    [0m[36;1m[1;3m          ID    age  area      assets  consumption_propensity ethnicity  \
    0          0  20t29   0.0    0.000000                    0.00  hispanic   
    1          1  20t29   0.0    0.000000                    0.00  hispanic   
    2          2  20t29   0.0    0.000000                    0.00     asian   
    3          3  20t29   0.0    0.000000                    0.00  hispanic   
    4          4  20t29   0.0    0.000000                    0.00  hispanic   
    ...      ...    ...   ...         ...                     ...       ...   
    56775  56775    U19  22.0  333.087189                    1.05     asian   
    56776  56776    U19  22.0  221.979263                    1.05     asian   
    56777  56777    U19  22.0  221.979263                    1.05  hispanic   
    56778  56778    U19  22.0  333.087189                    1.05  hispanic   
    56779  56779    U19  22.0  333.087189                    1.05  hispanic   
    
           gender  household  monthly_income  post_tax_income         region  \
    0        male        0.0        0.000000              0.0  Bay of Plenty   
    1        male        1.0        0.000000              0.0  Bay of Plenty   
    2      female        2.0        0.000000              0.0  Bay of Plenty   
    3        male        3.0        0.000000              0.0  Bay of Plenty   
    4        male        4.0        0.000000              0.0  Bay of Plenty   
    ...       ...        ...             ...              ...            ...   
    56775  female      450.0      196.094788              0.0     Wellington   
    56776  female      264.0      196.094788              0.0     Wellington   
    56777  female      234.0      196.094788              0.0     Wellington   
    56778    male      292.0      196.094788              0.0     Wellington   
    56779  female      309.0      196.094788              0.0     Wellington   
    
           will_work  month  year  adjusted_consumption  inflation_impact  
    0            0.0      0     0                   0.0          0.000000  
    1            0.0      0     0                   0.0          0.000000  
    2            0.0      0     0                   0.0          0.000000  
    3            0.0      0     0                   0.0          0.000000  
    4            0.0      0     0                   0.0          0.000000  
    ...          ...    ...   ...                   ...               ...  
    56775        0.0      4     0                   0.0        196.094788  
    56776        0.0      4     0                   0.0        196.094788  
    56777        0.0      4     0                   0.0        196.094788  
    56778        0.0      4     0                   0.0        196.094788  
    56779        0.0      4     0                   0.0        196.094788  
    
    [283900 rows x 16 columns][0m[32;1m[1;3mThe analysis of the simulation data shows that inflation has a varying impact on consumption behavior across different demographics. Here are some key observations:
    
    1. For younger age groups (20-29), the impact of inflation on consumption propensity is minimal. This could be due to their relatively lower assets and income levels.
    
    2. For the age group under 19, the impact of inflation is more pronounced. This could be due to their reliance on fixed income sources like allowances or part-time jobs, which do not adjust for inflation.
    
    3. Across different regions, the impact of inflation on consumption behavior also varies. For instance, in the Wellington region, the impact of inflation is more noticeable compared to the Bay of Plenty region.
    
    4. The impact of inflation on consumption behavior also varies by ethnicity and gender. For instance, the Asian and Hispanic communities show a higher impact of inflation on their consumption behavior compared to other ethnicities. Similarly, females show a higher impact of inflation on their consumption behavior compared to males.
    
    These observations suggest that inflation can affect consumption behavior in complex ways, depending on various demographic and economic factors.[0m
    
    [1m> Finished chain.[0m





    {'input': 'How is inflation affecting consumption behaviour?',
     'output': 'The analysis of the simulation data shows that inflation has a varying impact on consumption behavior across different demographics. Here are some key observations:\n\n1. For younger age groups (20-29), the impact of inflation on consumption propensity is minimal. This could be due to their relatively lower assets and income levels.\n\n2. For the age group under 19, the impact of inflation is more pronounced. This could be due to their reliance on fixed income sources like allowances or part-time jobs, which do not adjust for inflation.\n\n3. Across different regions, the impact of inflation on consumption behavior also varies. For instance, in the Wellington region, the impact of inflation is more noticeable compared to the Bay of Plenty region.\n\n4. The impact of inflation on consumption behavior also varies by ethnicity and gender. For instance, the Asian and Hispanic communities show a higher impact of inflation on their consumption behavior compared to other ethnicities. Similarly, females show a higher impact of inflation on their consumption behavior compared to males.\n\nThese observations suggest that inflation can affect consumption behavior in complex ways, depending on various demographic and economic factors.'}



#### Bring your own Population


```python
from models import macro_economics
from populations import NYC
from macro_economics.config import NYC_NUM_AGENTS

data_loader_nyc = DataLoader(model = macro_economics, region = NYC, population_size = NYC_NUM_AGENTS)
agents_sim = Executor(macro_economics,data_loader_nyc)
agents_sim.execute()
```

    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml
    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml
    Config saved at:  /Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml
    resolvers already registered..
    
    running episode 0...
    Substep Action: Earning decision
    LLM benchmark expts
    Some Example Prompts:  ['You are male of age 20t29, living in the BK region. It is September 2020, number of covid cases is 874.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are male of age 20t29, living in the BX region. It is September 2020, number of covid cases is 874.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are male of age 20t29, living in the MN region. It is September 2020, number of covid cases is 874.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ']
    Substep: Agent Earning!
    Substep: Agent Consumption
    Executing Substep: Labor Market
    Executing Substep: Financial Market
    Substep Action: Earning decision
    LLM benchmark expts
    Some Example Prompts:  ['You are male of age 20t29, living in the BK region. It is October 2020, number of covid cases is 701.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are male of age 20t29, living in the BX region. It is October 2020, number of covid cases is 701.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ', 'You are male of age 20t29, living in the MN region. It is October 2020, number of covid cases is 701.0. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures? ']
    Substep: Agent Earning!
    Substep: Agent Consumption
    Executing Substep: Labor Market
    Executing Substep: Financial Market
    Simulation Ended.

