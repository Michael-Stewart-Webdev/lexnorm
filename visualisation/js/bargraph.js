var graphColours = ["#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0", "#B276B2", "#F15854"]; // An array of colours that can be on the graph.

class BarGraph {

  constructor(x_axis_labels, graph_id) {

      var max = 1;
      var w = 1000;
      var h = 480;

      this.barColours = [graphColours.pop(), graphColours.pop()];
      
      var x0 = d3.scale.ordinal()
          .domain(d3.range(x_axis_labels.length))
          .rangeBands([0, w], .3);

      var x1 = d3.scale.ordinal()
          .domain(d3.range(2))
          .rangeBands([0, x0.rangeBand()]);

      var y = d3.scale.linear()
          .domain([0, max])
          .range([h, 0])
      var z = d3.scale.ordinal()
          //.range(["#661141", /*"#8a89a6",*/ "#3D1255", /*"#6b486b",*/ "#261758", /*"#d0743c",*/]);
          .range(this.barColours);
      // define the axis

      var xAxis = d3.svg.axis()
          .scale(x0)
          .orient("bottom")
          .tickFormat(function(d) { return x_axis_labels[d]; })
      var yAxis = d3.svg.axis()
          .scale(y)
          .orient("left")
          .ticks(10)
          .tickFormat(function(d) { return d ; });

      //setup the svg
      var svg = d3.select("#" + graph_id + "-svg")
          .attr("width", w+200)
          .attr("height", h+50)

      svg.append("svg:g")
          .attr("id", graph_id + "-barchart")
          .attr("transform", "translate(100,100)")
      
      var vis = d3.select("#" + graph_id + "-barchart");
      // x axis
      vis.append("g")          
          .attr("class", "x axis")
          .attr("transform", "translate(0," + (h) + ")")
          .call(xAxis)
        .selectAll("text")
          .data(x_axis_labels)
          .style("text-anchor", "middle")
          .attr("dx", "0px")
          .attr("dy", "15px")
          .text(function(d, i) { return d; });

      // x and y axis labels ("Method", "Percentage")
      vis.append("g")          
          .attr("transform", "translate(0," + (h) + ")")
        .append("text")
          .attr("y", 60)
          .attr("dx", w / 2 + 50)
          .attr("dy", "5px")
          .style("text-anchor", "end")
          .attr("class", "axis-label")
          .text("Method");
      vis.append("g")
          .attr("class", "y axis")
          .call(yAxis)
        .append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", -80)
          .attr("dx", - h / 2)
          .attr("dy", "5px")
          .style("text-anchor", "middle")
          .attr("id", "y-axis-label")
          .attr("class", "axis-label")
          .text("Count");

    // y Axis Grid (grey lines)
    var yAxisGrid = d3.svg.axis()
      .scale(y)
      .orient("left")
      .ticks(10)
      .tickSize(-w)
      .tickFormat("")
    vis.append("g")
      .attr("class", "y-axis-grid")
      .call(yAxisGrid)



    this.max = max;
    this.w = w;
    this.h = h;       
    this.x0 = x0;
    this.x1 = x1;
    this.y = y;
    this.z = z;
    this.xAxis = xAxis;
    this.yAxis = yAxis;
    this.yAxisGrid = yAxisGrid;
    this.svg = svg;
    this.vis = vis;
    this.graph_id = graph_id;

    this.addLegend();

  }

  addLegend() {

    var colors = this.barColours;

    
    var vis = this.vis
    var keys = ["Correct", "Tokens"];
    var graph_id = this.graph_id;

    $("#" + graph_id + "-g-legend").remove();

    var legend = vis.append("g")
        .attr("id", "" + graph_id + "-g-legend")
        .attr("text-anchor", "end")
        .attr("transform", "translate(0, -73)")
      .selectAll("g")
      .data(keys.slice().reverse())
      .enter().append("g")
        .attr("id", function(d, i) { return graph_id + "-legend-" + (i+1); })
        .attr("class", "legend")
        .attr("transform", function(d, i) { return "translate(0," + i * 26 + ")"; });

    legend.append("rect")
        .attr("x", this.w - 19)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", function(d, i) { return colors[i] });

    legend.append("text")
        .attr("x", this.w - 24)
        .attr("y", 9.5)
        .attr("dy", "0.32em")
        .text(function(d) { return d; });
  }

  update(data, numberOfTokensTotal) {

    var t = this;
    //addLegend();

      var vis = this.vis;

      var graph_id = this.graph_id;


      this.y.domain([0, d3.max(data, function(d) { return d[1]; })]);

      this.svg.select(".y")
        .transition()
        .call(t.yAxis);

      //a good written tutorial of d3 selections coming from protovis
      //http://www.jeromecukier.net/blog/2011/08/09/d3-adding-stuff-and-oh-understanding-selections/
      var bars_percent_tokens = vis.selectAll("rect.bar." + graph_id + "-bar-percent_tokens")
          .data(data);
    
      var bars_percent_correct = vis.selectAll("rect.bar." + graph_id + "-bar-percent_correct")
          .data(data);


      $("text.label." + graph_id + "-label-percent_tokens").remove();
      $("text.label." + graph_id + "-label-percent_correct").remove();

      var labels_percent_tokens = vis.selectAll("text.label." + graph_id + "-label-percent_tokens")
        .data(data)

      labels_percent_tokens.enter()
        .append("text")
        .attr("class", "label " + graph_id + "-label-percent_tokens")
        .attr("text-anchor", "middle")
        .attr("transform", function(d,i) {
          return "translate(" + [t.x0(i) + (t.x1.rangeBand()/2), 0] + ")"
        })
        .attr("y", function(d) { return t.y(d[1]) - 10; })      
        .text(function(d) { return d[1] > 0 ? d[1] : ""; });

      var labels_percent_correct = vis.selectAll("text.label." + graph_id + "-label-percent_correct")
        .data(data)

      labels_percent_correct.enter()
        .append("text")
        .attr("class", "label " + graph_id + "-label-percent_correct")
        .attr("text-anchor", "middle")
        .attr("transform", function(d,i) {
          return "translate(" + [t.x0(i) + (t.x1.rangeBand()/2) + t.x1.rangeBand(), 0] + ")"
        })
        .attr("y", function(d) { return t.y(d[2]) - 28; })      
        .text(function(d) { return d[2] > 0 ? d[2] : ""; })

      labels_percent_correct.enter() 
        .append("text")
		.attr("class", "label-sm label " + graph_id + "-label-percent_correct")
        .attr("text-anchor", "middle")
        .attr("transform", function(d,i) {
          return "translate(" + [t.x0(i) + (t.x1.rangeBand()/2) + t.x1.rangeBand(), 0] + ")"
        })
        .attr("y", function(d) { return t.y(d[2]) - 10; })   
        .text(function(d) { return d[2] > 0 ? "(" + (d[2]/d[1]*100).toFixed(1) + "%)" : ""; });


      bars_percent_tokens.enter()
          .append("svg:rect")
          .attr("class", "bar " + graph_id + "-bar-percent_tokens")
          .style("fill", t.z(2))
          .attr("width", t.x1.rangeBand() - 4)
          .attr("height", 0)
          .attr("y", t.h)
          .attr("transform", function(d,i) {
              return "translate(" + [t.x0(i) + 2, 0] + ")"
          })
          .on("mouseenter", function(){ 
            //maxLine
            //  .attr("class", "line maxline show")
            //  .attr("y1", y(maxP))
            //  .attr("y2", y(maxP))
            //  .style("stroke", z(2))
            $("." + graph_id + "-label-percent_tokens").addClass("faded-in");
            $("." + graph_id + "-label-percent_correct").addClass("faded-in");
            //$(".bar." + graph_id + "-bar-percent_correct").addClass("faded-out");
            //$("#" + t.graph_id + "-legend-2").addClass("faded-out");
          })
          .on("mouseleave", function(){ 
            $("." + graph_id + "-label-percent_tokens").removeClass("faded-in");
            $("." + graph_id + "-label-percent_correct").removeClass("faded-in");
            //maxLine.attr("class", "line maxline");
          });        
      bars_percent_correct.enter()
          .append("svg:rect")
          .attr("class", "bar " + graph_id + "-bar-percent_correct")
          .style("fill", t.z(1))
          .attr("width", t.x1.rangeBand() - 4)
          .attr("height", 0)
          .attr("y", t.h)
          .attr("transform", function(d,i) {
              return "translate(" + [t.x0(i) + 2 + t.x1.rangeBand(), 0] + ")"
          }) 
          .on("mouseenter", function(){ 
            //maxLine
            //  .attr("class", "line maxline show")
            //  .attr("y1", y(maxR))
            //  .attr("y2", y(maxR))
            //  .style("stroke", z(1))

            $("." + graph_id + "-label-percent_tokens").addClass("faded-in");
            $("." + graph_id + "-label-percent_correct").addClass("faded-in");

          })
          .on("mouseleave", function(){ 
            $("." + graph_id + "-label-percent_tokens").removeClass("faded-in");
            $("." + graph_id + "-label-percent_correct").removeClass("faded-in");
            //maxLine.attr("class", "line maxline");

          });

      bars_percent_tokens.exit()
      .transition()
      .duration(300)
      .ease("exp")
          .attr("height", 0)
          .remove()
      bars_percent_correct.exit()
      .transition()
      .duration(300)
      .ease("exp")
          .attr("height", 0)
          .remove()

      bars_percent_tokens
        .transition()
        .duration(300)
        .ease("quad")
            .attr("height", function(d, i) { return t.h - t.y(d[1])})
            .attr("y", function(d) { return t.y(d[1]); })

      bars_percent_correct
        .transition()
        .duration(300)
        .ease("quad")
            .attr("height", function(d, i) { return t.h - t.y(d[2])})
            .attr("y", function(d) { return t.y(d[2]); })


  }
}