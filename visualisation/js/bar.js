//Simple d3.js barchart example to illustrate d3 selections

//other good related tutorials
//http://www.recursion.org/d3-for-mere-mortals/
//http://mbostock.github.com/d3/tutorial/bar-1.html


var w = 1000
var h = 650

function addLegend() {

  var colors = ["#1b9e77", "#d95f02", "#7570b3"]

  console.log("added legend")
  $("#g-legend").remove();
  var vis = d3.select("#barchart");
  var keys = [measure_name, "Recall", "Precision"];


  legend = vis.append("g")
      .attr("id", "g-legend")
      .attr("text-anchor", "end")
      .attr("transform", "translate(0, -73)")
    .selectAll("g")
    .data(keys.slice().reverse())
    .enter().append("g")
      .attr("id", function(d, i) { return "legend-" + i; })
      .attr("class", "legend")
      .attr("transform", function(d, i) { return "translate(0," + i * 26 + ")"; });

  legend.append("rect")
      .attr("x", w - 19)
      .attr("width", 19)
      .attr("height", 19)
      .attr("fill", function(d, i) { return colors[i] });

  legend.append("text")
      .attr("x", w - 24)
      .attr("y", 9.5)
      .attr("dy", "0.32em")
      .text(function(d) { return d; });
}



function bars(data)
{

    addLegend();

    var vis = d3.select("#barchart")

    var colours = ["#6af784", "#6bf7ff", "#d0b5ff", "#ffb7d7", "#ffe9b7"];
    var stroke_colours = ["#53c668", "#51c0c6", "#ae98d6", "#d396b1", "#ddca9f"];

    //a good written tutorial of d3 selections coming from protovis
    //http://www.jeromecukier.net/blog/2011/08/09/d3-adding-stuff-and-oh-understanding-selections/
    var bars_precisions = vis.selectAll("rect.bar.bar-precision")
        .data(data)

  
    var bars_recalls = vis.selectAll("rect.bar.bar-recall")
        .data(data)

    var bars_fscores = vis.selectAll("rect.bar.bar-f1score")
        .data(data)

    $("text.label").remove(); // It took like one hour to try figure out why the enter() function wasn't working properly.
                              // I gave up in the end...

    var labels_fscores = vis.selectAll("text.label.label-fscore")
      .data(data)

    labels_fscores.enter()
      .append("text")
      .attr("class", "label label-fscore")
      .attr("text-anchor", "middle")
      .attr("transform", function(d,i) {
        return "translate(" + [x0(i) + 2 + (x1.rangeBand() * 2) + 22, 0] + ")"
      })
      .attr("y", function(d) { return y(d[measure_name]) - 10; })      
      .text(function(d) { return d[measure_name].toFixed(4); });

    var labels_precisions = vis.selectAll("text.label.label-precision")
      .data(data)

    labels_precisions.enter()
      .append("text")
      .attr("class", "label label-precision")
      .attr("text-anchor", "middle")
      .attr("transform", function(d,i) {
        return "translate(" + [x0(i) + 2 + 22, 0] + ")"
      })
      .attr("y", function(d) { return y(d[measure_name_p]) - 10; })      
      .text(function(d) { return d[measure_name_p].toFixed(4); });

    var labels_recalls = vis.selectAll("text.label.label-recall")
      .data(data)

    labels_recalls.enter()
      .append("text")
      .attr("class", "label label-recall")
      .attr("text-anchor", "middle")
      .attr("transform", function(d,i) {
        return "translate(" + [x0(i) + 2 + (x1.rangeBand()) + 22, 0] + ")"
      })
      .attr("y", function(d) { return y(d[measure_name_r]) - 10; })      
      .text(function(d) { return d[measure_name_r].toFixed(4); });

    bars_precisions.enter()
        .append("svg:rect")
        .attr("class", "bar bar-precision")
        .style("fill", z(2))
        .attr("width", x1.rangeBand() - 4)
        .attr("height", 0)
        .attr("y", h)
        .attr("transform", function(d,i) {
            return "translate(" + [x0(i) + 2, 0] + ")"
        })
        .on("mouseenter", function(){ 
          //maxLine
          //  .attr("class", "line maxline show")
          //  .attr("y1", y(maxP))
          //  .attr("y2", y(maxP))
          //  .style("stroke", z(2))
          $(".label-precision").addClass("faded-in");
          $(".bar.bar-f1score").addClass("faded-out");
          $(".bar.bar-recall").addClass("faded-out");
          $("#legend-2").addClass("faded-out");
          $("#legend-1").addClass("faded-out");
        })
        .on("mouseleave", function(){ 
          $(".label-precision").removeClass("faded-in");
          $(".bar.bar-f1score").removeClass("faded-out");
          $(".bar.bar-recall").removeClass("faded-out");
          $("#legend-2").removeClass("faded-out");
          $("#legend-1").removeClass("faded-out");
          //maxLine.attr("class", "line maxline");
        });        
    bars_recalls.enter()
        .append("svg:rect")
        .attr("class", "bar bar-recall")
        .style("fill", z(1))
        .attr("width", x1.rangeBand() - 4)
        .attr("height", 0)
        .attr("y", h)
        .attr("transform", function(d,i) {
            return "translate(" + [x0(i) + 2 + x1.rangeBand(), 0] + ")"
        }) 
        .on("mouseenter", function(){ 
          //maxLine
          //  .attr("class", "line maxline show")
          //  .attr("y1", y(maxR))
          //  .attr("y2", y(maxR))
          //  .style("stroke", z(1))
          $(".label-recall").addClass("faded-in");
          $(".bar.bar-f1score").addClass("faded-out");
          $(".bar.bar-precision").addClass("faded-out");
          $("#legend-2").addClass("faded-out");
          $("#legend-0").addClass("faded-out");
        })
        .on("mouseleave", function(){ 
          $(".label-recall").removeClass("faded-in");
          $(".bar.bar-f1score").removeClass("faded-out");
          $(".bar.bar-precision").removeClass("faded-out");
          $("#legend-2").removeClass("faded-out");
          $("#legend-0").removeClass("faded-out");
          //maxLine.attr("class", "line maxline");

        });


    bars_fscores.enter()
        .append("svg:rect")
        .attr("class", "bar bar-f1score")
        .style("fill", z(0))
        .attr("width", x1.rangeBand() - 4)
        .attr("height", 0)
        .attr("y", h)
        .attr("transform", function(d,i) {
            return "translate(" + [x0(i) + 2 + (x1.rangeBand() * 2), 0] + ")"
        })
        .on("mouseenter", function(){ 
          //maxLine
          //  .attr("class", "line maxline show")
          //  .attr("y1", y(maxF))
          //  .attr("y2", y(maxF))
          //  .style("stroke", z(0))
          $(".label-fscore").addClass("faded-in");
          $(".bar.bar-recall").addClass("faded-out");
          $(".bar.bar-precision").addClass("faded-out");
          $("#legend-0").addClass("faded-out");
          $("#legend-1").addClass("faded-out");
        })
        .on("mouseleave", function(){ 
          $(".label-fscore").removeClass("faded-in");
          $(".bar.bar-recall").removeClass("faded-out");
          $(".bar.bar-precision").removeClass("faded-out");
          $("#legend-0").removeClass("faded-out");
          $("#legend-1").removeClass("faded-out");
          //maxLine.attr("class", "line maxline");
        });

    //exit 
    bars_fscores.exit()
    .transition()
    .duration(300)
    .ease("exp")
        .attr("height", 0)
        .remove()
    bars_precisions.exit()
    .transition()
    .duration(300)
    .ease("exp")
        .attr("height", 0)
        .remove()
    bars_recalls.exit()
    .transition()
    .duration(300)
    .ease("exp")
        .attr("height", 0)
        .remove()

    bars_fscores
    .transition()
    .duration(300)
    .ease("quad")
        .attr("height", function(d, i) { return h - y(d[measure_name])})
        .attr("y", function(d) { return y(d[measure_name]); })
    bars_precisions
    .transition()
    .duration(300)
    .ease("quad")
        .attr("height", function(d, i) { return h - y(d[measure_name_p])})
        .attr("y", function(d) { return y(d[measure_name_p]); })
    bars_recalls
    .transition()
    .duration(300)
    .ease("quad")
        .attr("height", function(d, i) { return h - y(d[measure_name_r])})
        .attr("y", function(d) { return y(d[measure_name_r]); })

}



function init_barChart()
{

    var data = barGraphData;

    max = 1

    x0 = d3.scale.ordinal()
        .domain(d3.range(data[0].length))
        .rangeBands([0, w], .3);

    x1 = d3.scale.ordinal()
        .domain(d3.range(3))
        .rangeBands([0, x0.rangeBand()]);

    y = d3.scale.linear()
        .domain([0, max])
        .range([h, 0])
    z = d3.scale.ordinal()
        //.range(["#661141", /*"#8a89a6",*/ "#3D1255", /*"#6b486b",*/ "#261758", /*"#d0743c",*/]);
        .range(["#1b9e77", /*"#8a89a6",*/ "#d95f02", /*"#6b486b",*/ "#7570b3", /*"#d0743c",*/]);


    // define the axis
    var xAxis = d3.svg.axis()
        .scale(x0)
        .orient("bottom")
        .tickFormat(function(d) { return data[d].exp_name; })


    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .ticks(10)
        .tickFormat(function(d) { return d; });

    //setup the svg
    var svg = d3.select("#svg")
        .attr("width", w+200)
        .attr("height", h+200)

    svg.append("svg:rect")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("stroke", "#000")
        .attr("fill", "none")

    svg.append("svg:g")
        .attr("id", "barchart")
        .attr("transform", "translate(100,100)")
    
    var vis = d3.select("#barchart");

    vis.append("g")          
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (h) + ")")
        .call(xAxis)
      .selectAll("text")
        .data(data)
        .style("text-anchor", "middle")
        .attr("dx", "0px")
        .attr("dy", "15px")
        .text(function(d, i) { return d[i].exp_name; });

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
        .text(measure_name);

  addLegend();

  var yAxisGrid = d3.svg.axis()
      .scale(y)
      .orient("left")
      .ticks(10)
      .tickSize(-w)
      .tickFormat("")

  vis.append("g")
      .attr("class", "y-axis-grid")
      .call(yAxisGrid)
    

}


function reloadBarGraph() {
   bars(barGraphData[current_set - 1]);
}