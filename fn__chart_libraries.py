from fn__libraries import *







########################################





help_text_01 = '''\
    For example, to see the dry bulb temperature above 10, you
    can use the conditional statement, "a>10" without quotes. To see dry bulb temperature between -5 and 10 you can use the
    conditional statement, "a>-5 and a<10" without quotes.
    '''


def absrd_apply_analysis_period(data, plot_analysis_period):
    start_month, start_day, start_hour, end_month, end_day, end_hour = plot_analysis_period
    lb_ap = AnalysisPeriod(start_month, start_day, start_hour, end_month, end_day, end_hour)
    data = data.filter_by_analysis_period(lb_ap)
    return data



def absrd_slice_df_analysis_period(df, plot_analysis_period):
    start_month, start_day, start_hour, end_month, end_day, end_hour = plot_analysis_period

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["datetime"])

    # Handle periods that wrap around the year
    if (start_month > end_month) or (start_month == end_month and start_day > end_day):
        # Filter for two periods: from start to end of the year and from start of the year to end period
        condition = (
            ((df.index.month > start_month) | 
             ((df.index.month == start_month) & (df.index.day >= start_day)) |
             ((df.index.month == start_month) & (df.index.day == start_day) & (df.index.hour >= start_hour))) |
            ((df.index.month < end_month) | 
             ((df.index.month == end_month) & (df.index.day <= end_day)) |
             ((df.index.month == end_month) & (df.index.day == end_day) & (df.index.hour <= end_hour)))
        )
    else:
        # Standard filter within the same year
        condition = (
            (df.index.month > start_month) | 
            ((df.index.month == start_month) & (df.index.day >= start_day)) |
            ((df.index.month == start_month) & (df.index.day == start_day) & (df.index.hour >= start_hour))
        ) & (
            (df.index.month < end_month) | 
            ((df.index.month == end_month) & (df.index.day <= end_day)) |
            ((df.index.month == end_month) & (df.index.day == end_day) & (df.index.hour <= end_hour))
        )

    # Apply the condition to slice the DataFrame
    filtered_df = df[condition]

    return filtered_df







def absrd_avg_daily_profile(plot_data, plot_analysis_period, global_colorset):
    """Return the daily profile based on the 'plot_data'."""
    var_name = str(plot_data.header.data_type)
    var_unit = str(plot_data.header.unit)
    
    # Set maximum and minimum according to data
    data_max = 5 * ceil(plot_data.max / 5)
    data_min = 5 * floor(plot_data.min / 5)
    range_y = [data_min, data_max]

    # Create a DataFrame with appropriate datetime information
    df = pd.DataFrame(plot_data.values, columns=["value"])
    df["datetime"] = pd.date_range(start="2023-01-01 00:00", periods=len(df), freq="H")
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    
    # Apply the slicing function to filter data according to the analysis period
    df = absrd_slice_df_analysis_period(df, plot_analysis_period)
    
    # Check if the resulting DataFrame is empty after slicing
    if df.empty:
        st.warning("No data available for the selected period. Please adjust your filtering criteria.")
        return go.Figure()  # Return an empty figure to avoid errors
    
    # Calculate monthly averages
    var_month_ave = df.groupby(["month", "hour"])["value"].median().reset_index()
    
    # Normalize the data to map to colors
    var_color = colorsets[global_colorset]
    norm = (df["value"] - df["value"].min()) / (df["value"].max() - df["value"].min())
    df["color"] = norm.apply(lambda x: rgb_to_hex(var_color[int(x * (len(var_color) - 1))][:3]))

    # Get the unique months present in the sliced data
    unique_months = df["month"].unique()
    unique_months.sort()  # Ensure months are in order

    # Check if there are any months to plot
    if len(unique_months) == 0:
        st.warning("No data available for the selected period. Please adjust your filtering criteria.")
        return go.Figure()  # Return an empty figure if no months are found

    # Create subplots for each month present in the data
    fig = make_subplots(
        rows=1,
        cols=len(unique_months),
        shared_yaxes=True,
        subplot_titles=[pd.to_datetime(str(m), format='%m').strftime('%b') for m in unique_months],
        horizontal_spacing=0.01,  # Adjust horizontal spacing here
    )

    # Plot data for each unique month
    for i, month in enumerate(unique_months):
        monthly_data = df[df["month"] == month]
        monthly_avg = var_month_ave[var_month_ave["month"] == month]

        fig.add_trace(
            go.Scatter(
                x=monthly_data["hour"],
                y=monthly_data["value"],
                mode="markers",
                marker=dict(color=monthly_data["color"], opacity=0.5, size=3),
                opacity=1,
                name=pd.to_datetime(str(month), format='%m').strftime('%b'),
                showlegend=False,
                customdata=monthly_data["month"],
                hovertemplate=(
                    f"<b>{var_name}: {{%y:.2f}} {var_unit}</b><br>"
                    "Month: %{customdata}<br>"
                    "Hour: %{x}:00<br>"
                ),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_avg["hour"],
                y=monthly_avg["value"],
                mode="lines",
                line_color='black',
                line_width=2.5,
                showlegend=False,
                hovertemplate=(
                    f"<b>{var_name}: {{%y:.2f}} {var_unit}</b><br>"
                    "Hour: %{x}:00<br>"
                ),
            ),
            row=1,
            col=i + 1,
        )

        fig.update_xaxes(range=[0, 24], row=1, col=i + 1)
        fig.update_yaxes(range=range_y, row=1, col=i + 1)

    fig.update_xaxes(
        ticktext=["6", "12", "18"], tickvals=[6, 12, 18], tickangle=0
    )

    fig.update_layout(
        template='plotly_white',
        dragmode=False,
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        title=f"{var_name} ({var_unit})",
    )

    return fig





def plot_hourly_line_chart_with_slider(df: pd.DataFrame, variable: str, global_colorset: str):
    """
    Plot hourly line chart with range sliders.

    Args:
        df: DataFrame with hourly data to be plotted.
        variable: The variable name to be plotted.
        global_colorset: The colorset name to use for the plot.

    Returns:
        A Streamlit plotly figure with range sliders.
    """

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["datetime"])

    # Create initial figure
    fig = go.Figure()

    # Normalize the data to map to colors
    var_color = colorsets[global_colorset]
    norm = (df[variable] - df[variable].min()) / (df[variable].max() - df[variable].min())
    df["color"] = norm.apply(lambda x: rgb_to_hex(var_color[int(x * (len(var_color) - 1))][:3]))

    # Add line plot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[variable],
            mode="lines",
            line=dict(color=df["color"].iloc[0], width=2),
            name=variable,
            showlegend=True,
        )
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        title=f"Hourly Line Chart for {variable}",
        xaxis_title="Time",
        yaxis_title=variable,
        template='plotly_white',
        height=500,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    # Use Streamlit range slider to interactively filter the data
    start_date, end_date = st.slider(
        "Select Date Range",
        value=(df.index.min(), df.index.max()),
        format="YYYY-MM-DD HH:mm"
    )

    # Filter data based on the slider selection
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]

    # Update the figure with filtered data
    fig.update_traces(
        x=filtered_df.index,
        y=filtered_df[variable],
        selector=dict(name=variable)
    )

    st.plotly_chart(fig)

# Example usage:
# Assuming `data` is a DataFrame containing your hourly data and "value" is the column to be plotted.
# plot_hourly_line_chart_with_slider(data, "value", "original")














def slice_data_by_month(plot_data, start_month, end_month):
    
    df = pd.DataFrame(plot_data.values, columns=["value"])
    df["datetime"] = pd.date_range(start="2023-01-01 00:00", periods=len(df), freq="H")
    df["month"] = df["datetime"].dt.month

    # Filter the DataFrame based on the start and end months
    sliced_df = df[(df['month'] >= start_month) & (df['month'] <= end_month)]
    return sliced_df




def bin_timeseries_data(df, min_val, max_val, step):
    # Create bins based on the specified min, max, and step values
    bins = np.arange(min_val, max_val + step, step)
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    
    # Bin the 'value' column
    df['binned'] = pd.cut(df['value'], bins=bins, labels=labels, right=False)
    
    # Group by 'month' and 'binned', and count the occurrences
    binned_data = df.groupby(['month', 'binned']).size().reset_index(name='count')
    
    return binned_data



def normalize_data(binned_data):
    # Calculate the total hours per month
    month_totals = binned_data.groupby('month')['count'].sum().reset_index(name='total')
    # Merge the totals back into the binned data
    binned_data = binned_data.merge(month_totals, on='month')
    # Calculate the percentage for each bin
    binned_data['percentage'] = (binned_data['count'] / binned_data['total']) * 100
    return binned_data





import calendar

def create_stacked_bar_chart(binned_data, color_map, normalize):
    # Ensure the 'binned' column is of string type for Plotly
    binned_data['binned'] = binned_data['binned'].astype(str)

    # Map month numbers to three-letter month abbreviations
    binned_data['month'] = binned_data['month'].apply(lambda x: calendar.month_abbr[int(x)])

    # Determine the value field based on normalization
    value_field = 'percentage' if normalize else 'count'

    # Normalize the data to map to colors
    var_color = colorsets[color_map]
    df = binned_data
    df['bin_min'] = df['binned'].apply(lambda x: x.split('-')[0]).astype(int)
    norm = (df["bin_min"] - df["bin_min"].min()) / (df["bin_min"].max() - df["bin_min"].min())
    df["color"] = norm.apply(lambda x: rgb_to_hex(var_color[int(x * (len(var_color) - 1))][:3]))

    # Get unique months and bins
    months = binned_data['month'].unique()
    bins = binned_data['binned'].unique()

    # Create the stacked bar chart using Plotly
    fig = go.Figure()

    # Create a trace for each bin
    for bin_value in bins:
        bin_data = binned_data[binned_data['binned'] == bin_value]

        fig.add_trace(go.Bar(
            x=bin_data['month'],
            y=bin_data[value_field],
            name=bin_value,
            marker_color=bin_data['color'].tolist()
        ))

    yaxis_title = "Percentage of Hours (%)" if normalize else "Total Hours"

    # Update layout to ensure x-axis labels are treated as categorical text
    fig.update_layout(
        barmode='stack',
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        height=450,
        margin=dict(l=40, r=40, t=5, b=10),
    )

    # Ensure x-axis labels are treated as text categories
    fig.update_xaxes(type='category')

    return fig










def get_figure_config(title: str) -> dict:
    """Set figure config so that a figure can be downloaded as SVG."""

    return {
        'toImageButtonOptions': {
            'format': 'svg',  # one of png, svg, jpeg, webp
            'filename': title,
            'height': 350,
            'width': 700,
            'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
        }
    }



@st.cache_data
def get_fields() -> dict:
    # A dictionary of EPW variable name to its corresponding field number
    return {EPWFields._fields[i]['name'].name: i for i in range(6, 34)}


def get_diurnal_average_chart_figure(epw: EPW, global_colorset: str, switch: bool = False) -> Figure:
    """Create a diurnal average chart from EPW.

    Args:
        epw: An EPW object.
        global_colorset: A string representing the name of a Colorset.

    Returns:
        A plotly figure.
    """
    colors = get_colors(switch, global_colorset)
    return epw.diurnal_average_chart(show_title=True, colors=colors)


def get_hourly_data_figure(
        data: HourlyContinuousCollection, global_colorset: str, conditional_statement: str,
        min: float, max: float, st_month: int, st_day: int, st_hour: int, end_month: int,
        end_day: int, end_hour: int) -> Figure:
    """Create heatmap from hourly data.

    Args:
        data: HourlyContinuousCollection object.
        global_colorset: A string representing the name of a Colorset.
        conditional_statement: A string representing a conditional statement.
        min: A string representing the lower bound of the data range.
        max: A string representing the upper bound of the data range.
        st_month: start month.
        st_day: start day.
        st_hour: start hour.
        end_month: end month.
        end_day: end day.
        end_hour: end hour.

    Returns:
        A plotly figure.
    """
    lb_ap = AnalysisPeriod(st_month, st_day, st_hour, end_month, end_day, end_hour)
    data = data.filter_by_analysis_period(lb_ap)

    if conditional_statement:
        try:
            data = data.filter_by_conditional_statement(
                conditional_statement)
        except AssertionError:
            return 'No values found for that conditional statement'
        except ValueError:
            return 'Invalid conditional statement'

    if min:
        try:
            min = float(min)
        except ValueError:
            return 'Invalid minimum value'

    if max:
        try:
            max = float(max)
        except ValueError:
            return 'Invalid maximum value'

    lb_lp = LegendParameters(colors=colorsets[global_colorset])

    if min:
        lb_lp.min = min
    if max:
        lb_lp.max = max

    hourly_plot = HourlyPlot(data, legend_parameters=lb_lp)

    # AB code tweaks
    var_name = str(data.header.data_type)
    var_unit = str(data.header.unit)
    var_plot_title = f'{var_name} ({var_unit})'

    # Inspect available methods and attributes
    # st.write( dir(hourly_plot) )

    return hourly_plot.plot(title=var_plot_title, show_title=True)







def get_bar_chart_figure(fields: dict, epw: EPW, selection: List[str], data_type: str,
                         switch: bool, stack: bool, global_colorset: str) -> Figure:
    """Create bar chart figure.

    Args:
        fields: A dictionary of EPW variable name to its corresponding field number.
        epw: An EPW object.
        selection: A list of strings representing the names of the fields to be plotted.
        data_type: A string representing the data type of the data to be plotted.
        switch: A boolean to indicate whether to reverse the colorset.
        stack: A boolean to indicate whether to stack the bars.
        global_colorset: A string representing the name of a Colorset.

    Returns:
        A plotly figure.
    """
    colors = get_colors(switch, global_colorset)

    data = []
    for count, item in enumerate(selection):
        if item:
            var = epw._get_data_by_field(fields[list(fields.keys())[count]])
            if data_type == 'Monthly average':
                data.append(var.average_monthly())
            elif data_type == 'Monthly total':
                data.append(var.total_monthly())
            elif data_type == 'Daily average':
                data.append(var.average_daily())
            elif data_type == 'Daily total':
                data.append(var.total_daily())

    lb_lp = LegendParameters(colors=colors)
    monthly_chart = MonthlyChart(
        data, legend_parameters=lb_lp, stack=stack
    )
    return monthly_chart.plot(title=data_type, center_title=True)




# # Following code is hashed as I have created code with greater functionality for font size - family and layout
# def get_hourly_line_chart_figure(data: HourlyContinuousCollection,
#                                  switch: bool, global_colorset: str) -> Figure:
#     """Create hourly line chart figure.

#     Args:
#         data: An HourlyContinuousCollection object.
#         switch: A boolean to indicate whether to reverse the colorset.
#         global_colorset: A string representing the name of a Colorset.

#     Returns:
#         A plotly figure.
#     """
#     colors = get_colors(switch, global_colorset)
#     return data.line_chart(
#         color=colors[-1], title=data.header.data_type.name, show_title=True
#     )


def get_hourly_line_chart_figure(
    data: HourlyContinuousCollection, switch: bool, global_colorset: str, 
    font_family: str = "SansSerif", font_size: int = 12, margin: dict = None) -> Figure:
    """
    Create hourly line chart figure with custom font and layout options.

    Args:
        data: An HourlyContinuousCollection object.
        switch: A boolean to indicate whether to reverse the colorset.
        global_colorset: A string representing the name of a Colorset.
        font_family: The font family to use for the chart text.
        font_size: The size of the font to use for the chart text.
        margin: A dictionary specifying margin values for the chart layout (e.g., {'l': 20, 'r': 20, 't': 20, 'b': 20}).

    Returns:
        A customized Plotly figure.
    """
    # Set default margin if not provided
    if margin is None:
        margin = dict(l=20, r=20, t=30, b=10)

    # Get the colors for the plot
    colors = get_colors(switch, global_colorset)

    # Create the initial line chart using Ladybug's method
    figure = data.line_chart(
        color=colors[-1], title=data.header.data_type.name, show_title=True
    )

    # Customize the layout of the figure
    figure.update_layout(
        # font=dict(family=font_family, size=font_size),
        margin=margin,
        title=dict(
            x=0.5,  # Center the title
            xanchor='center',
            # font=dict(size=font_size+2,family=font_family)
        ),
        # xaxis=dict(
        #     title_font=dict(family=font_family, size=font_size),
        #     tickfont=dict(family=font_family, size=font_size)
        # ),
        # yaxis=dict(
        #     title_font=dict(family=font_family, size=font_size),
        #     tickfont=dict(family=font_family, size=font_size)
        # ),
    )

    # Customize legend properties if present
    if figure.layout.legend:
        figure.update_layout(
            legend=dict(
                # font=dict(family=font_family, size=font_size),
                orientation="h",  # Horizontal legend
                x=0.5,
                xanchor="center",
                y=-0.1,  # Position the legend below the plot
                yanchor="top"
            )
        )

    return figure





def get_hourly_diurnal_average_chart_figure(data: HourlyContinuousCollection,
                                            switch: bool, global_colorset: str) -> Figure:
    """Create diurnal average chart figure for hourly data.

    Args:
        data: An HourlyContinuousCollection object.
        switch: A boolean to indicate whether to reverse the colorset.
        global_colorset: A string representing the name of a Colorset.

    Returns:
        A plotly figure.
    """
    colors = get_colors(switch, global_colorset)
    return data.diurnal_average_chart(
        title=data.header.data_type.name, show_title=True,
        color=colors[-1])





def get_daily_chart_figure(data: HourlyContinuousCollection, switch: bool,
                           global_colorset: str) -> Figure:
    """Create daily chart figure.

    Args:
        data: An HourlyContinuousCollection object.
        switch: A boolean to indicate whether to reverse the colorset.
        global_colorset: A string representing the name of a Colorset.

    Returns:
        A plotly figure.
    """
    colors = get_colors(switch, global_colorset)
    data = data.average_daily()

    return data.bar_chart(color=colors[-1], title=data.header.data_type.name,
                          show_title=True)





def get_sunpath_figure(sunpath_type: str, global_colorset: str, epw: EPW = None,
                       switch: bool = False,
                       data: HourlyContinuousCollection = None, ) -> Figure:
    """Create sunpath figure.

    Args:
        sunpath_type: A string representing the type of sunpath to be plotted.
        global_colorset: A string representing the name of a Colorset.
        epw: An EPW object.
        switch: A boolean to indicate whether to reverse the colorset.
        data: Hourly data to load on sunpath.


    Returns:
        A plotly figure.
    """
    if sunpath_type == 'from epw location':
        lb_sunpath = Sunpath.from_location(epw.location)
        colors = get_colors(switch, global_colorset)
        title = epw.location.city
        return lb_sunpath.plot(colorset=colors, title=title, show_title=True)
    else:
        lb_sunpath = Sunpath.from_location(epw.location)
        colors = colorsets[global_colorset]
        title = data.header.data_type.name
        return lb_sunpath.plot(colorset=colors, data=data, title=title, show_title=True)




def get_degree_days_figure(
    dbt: HourlyContinuousCollection, _heat_base_: int, _cool_base_: int,
    stack: bool, switch: bool, global_colorset: str) -> Tuple[Figure,
                                                              HourlyContinuousCollection,
                                                              HourlyContinuousCollection]:
    """Create HDD and CDD figure.

    Args:
        dbt: A HourlyContinuousCollection object.
        _heat_base_: A number representing the heat base temperature.
        _cool_base_: A number representing the cool base temperature.
        stack: A boolean to indicate whether to stack the data.
        switch: A boolean to indicate whether to reverse the colorset.
        global_colorset: A string representing the name of a Colorset.

    Returns:
        A tuple of three items:

        -   A plotly figure.

        -   Heating degree days as a HourlyContinuousCollection.

        -   Cooling degree days as a HourlyContinuousCollection.
    """

    hourly_heat = HourlyContinuousCollection.compute_function_aligned(
        heating_degree_time, [dbt, _heat_base_],
        HeatingDegreeTime(), 'degC-hours')
    hourly_heat.convert_to_unit('degC-days')

    hourly_cool = HourlyContinuousCollection.compute_function_aligned(
        cooling_degree_time, [dbt, _cool_base_],
        CoolingDegreeTime(), 'degC-hours')
    hourly_cool.convert_to_unit('degC-days')

    colors = get_colors(switch, global_colorset)

    lb_lp = LegendParameters(colors=colors)
    monthly_chart = MonthlyChart(
        [hourly_cool.total_monthly(), hourly_heat.total_monthly()],
        legend_parameters=lb_lp, stack=stack)

    return monthly_chart.plot(title='Degree Days', center_title=True), hourly_heat, hourly_cool


def get_windrose_figure(st_month: int, st_day: int, st_hour: int, end_month: int,
                        end_day: int, end_hour: int, epw, global_colorset) -> Figure:
    """Create windrose figure.

    Args:
        st_month: A number representing the start month.
        st_day: A number representing the start day.
        st_hour: A number representing the start hour.
        end_month: A number representing the end month.
        end_day: A number representing the end day.
        end_hour: A number representing the end hour.
        epw: An EPW object.
        global_colorset: A string representing the name of a Colorset.

    Returns:
        A plotly figure.
    """

    lb_ap = AnalysisPeriod(st_month, st_day, st_hour, end_month, end_day, end_hour)
    wind_dir = epw.wind_direction.filter_by_analysis_period(lb_ap)
    wind_spd = epw.wind_speed.filter_by_analysis_period(lb_ap)

    lb_lp = LegendParameters(colors=colorsets[global_colorset])
    lb_wind_rose = WindRose(wind_dir, wind_spd)
    lb_wind_rose.legend_parameters = lb_lp

    return lb_wind_rose.plot(title='Wind-Rose', show_title=True)




def get_psy_chart_figure(epw: EPW, global_colorset: str, selected_strategy: str,
                         load_data: bool, draw_polygons: bool,
                         data: HourlyContinuousCollection) -> Figure:
    """Create psychrometric chart figure.

    Args:
        epw: An EPW object.
        global_colorset: A string representing the name of a Colorset.
        selected_strategy: A string representing the name of a psychrometric strategy.
        load_data: A boolean to indicate whether to load the data.
        draw_polygons: A boolean to indicate whether to draw the polygons.
        data: Hourly data to load on psychrometric chart.

    Returns:
        A plotly figure.
    """

    lb_lp = LegendParameters(colors=colorsets[global_colorset])
    lb_psy = PsychrometricChart(epw.dry_bulb_temperature,
                                epw.relative_humidity, legend_parameters=lb_lp)

    if selected_strategy == 'All':
        strategies = [Strategy.comfort, Strategy.evaporative_cooling,
                      Strategy.mas_night_ventilation, Strategy.occupant_use_of_fans,
                      Strategy.capture_internal_heat, Strategy.passive_solar_heating]
    elif selected_strategy == 'Comfort':
        strategies = [Strategy.comfort]
    elif selected_strategy == 'Evaporative Cooling':
        strategies = [Strategy.evaporative_cooling]
    elif selected_strategy == 'Mass + Night Ventilation':
        strategies = [Strategy.mas_night_ventilation]
    elif selected_strategy == 'Occupant use of fans':
        strategies = [Strategy.occupant_use_of_fans]
    elif selected_strategy == 'Capture internal heat':
        strategies = [Strategy.capture_internal_heat]
    elif selected_strategy == 'Passive solar heating':
        strategies = [Strategy.passive_solar_heating]

    pmv = PolygonPMV(lb_psy)

    if load_data:
        if draw_polygons:
            figure = lb_psy.plot(data=data, polygon_pmv=pmv,
                                 strategies=strategies,
                                 solar_data=epw.direct_normal_radiation,)
        else:
            figure = lb_psy.plot(data=data)
    else:
        if draw_polygons:
            figure = lb_psy.plot(polygon_pmv=pmv, strategies=strategies,
                                 solar_data=epw.direct_normal_radiation)
        else:
            figure = lb_psy.plot()

    return figure








