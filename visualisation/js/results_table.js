    class ResultsTable {


      constructor() {

        this.button = $("#results-table-button a");
        this.button.on("click", this.showResultsTable);

        function shortName(method) {
          var sn = "";
          var m = method.match(/\_./g);
          sn += method.substring(0, method.indexOf("_")) + "_";
          for(var i in m) {
            sn += m[i][1];
          }
          return sn;
        }

        function generateTdForScore(score, currentMax) {
          var td_str = "";
          var good_color = "rgba(106,247,132" // good
          var bad_color = "rgba(247,118,136" // bad
          var good_color_border = "rgba(83,204,105";
          var bad_color_border = "rgba(211,101,116";

          if(score < 0.6) {
            var col = bad_color;
            var col_b = bad_color_border;
            var alpha = (1-(score-0.1))**5;
          } else {
            var col = good_color;
            var col_b = good_color_border;
            var alpha = (score+0.1)**5;
          }
          if(score == currentMax) {
            td_str = "<td style=\"background: " + col + "," + alpha + ")" + "; border-color: " + col_b + "," + alpha + ")\"><b>" + score.toFixed(4) + "</b></td>";
          } else {
            td_str = "<td style=\"background: " + col + "," + alpha + ")" + "; border-color: " + col_b + "," + alpha + ")\">" + score.toFixed(4) + "</td>";
          }
          return td_str;
          
        }

        // Headers
        var rth = $("#results-table thead");    
        var th_str = "";
        var firstSetMethods = [];
        for(var dataset in results) {
          th_str += "<th>Dataset</th>"
          for(var method in results[dataset]) {
            th_str += "<th title=\"" + method + "\">" + shortName(method) + "</th>";
            firstSetMethods.push(method);
          }
          break;
        }
        rth.append("<tr>" + th_str + "</tr>");

        var rtt = $("#results-table tbody")
        $("#results-table-tbody tr").remove();




        for(var dataset in results) {
          var td_str = "<td>" + dataset + "</td>";

          var currentMax = 0.0;
          for(var method in firstSetMethods) {
            if(results[dataset][firstSetMethods[method]]) {
              if(results[dataset][firstSetMethods[method]]["scores"]["F1"] > currentMax) {
                currentMax = results[dataset][firstSetMethods[method]]["scores"]["F1"];
              }
            }
          }


          for(var method in firstSetMethods) { 
            if(results[dataset][firstSetMethods[method]]) {
              td_str += generateTdForScore(results[dataset][firstSetMethods[method]]["scores"]["F1"], currentMax);
            } else {
              td_str += "<td>-</td>"
            }            
          }
          rtt.append("<tr>" + td_str + "</tr>");
        }


        $('#results-table td, #results-table th').hover(function() { 
          $("#results-table").toggleClass("hover");
          var i = $(this).index();
          $(this).parents("tr").toggleClass("hover");
          $("#results-table td:nth-child(" + (i+1) + ")").toggleClass("hover");
          $("#results-table th:nth-child(" + (i+1) + ")").toggleClass("hover");
        });



        $('#results-table td').on('click', function() {

          var row = $(this).parents("tr").index();
          var col = $(this).index();

          if(col > 0) {
            $("#sets-ul li:nth-child(" + (row+1) + ") a").click()
            clickMethodWithIndex(col-1);
          }
        });

      }

      hideResultsTable() {
        nav_sets.removeClass("hide");
        $("#results-table").hide();
        $("#results-table-button").removeClass("active");
        $("#results-table").hide();
          $("#results-table-wrapper").hide();
        $("#method-results-wrapper").show();
      }

      showResultsTable() {
        nav_sets.addClass("hide");
          $("#results-table").show();
          $(this).parent().addClass("active");
          $("#results-table-wrapper").show();
        $("#method-results-wrapper").hide();
      }
    }