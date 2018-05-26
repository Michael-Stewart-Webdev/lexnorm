class Method {
  constructor(name, normalised_tokens, normalised_documents, scores) {

    if(!normalised_tokens || !scores) {
      throw new Error("Method " + name + " must have normalised tokens and scores to be visualised.")
    }

    this.name = name;
    this.normalised_tokens = normalised_tokens;
    // No need to store normalised documents
    this.scores = scores;

    var ls = document.createElement("LI");
    ls.classList.add("hidden");
    var lsa = document.createElement("A");
    
    lsa.innerHTML = this.name;
    ls.appendChild(lsa);
    method_dropdown_items[0].appendChild(ls);

    this.button = $(ls);

    this.hide();
    var t = this;
    this.button.on('click', function() {
      t.displayResults()
    });

    var gd =  this.getGraphData();
    this.tokenClassGraphData = gd[0];
    this.sourceGraphData = gd[1];
    this.classifierPredictionsGraphData = gd[2];






    this.table_html = ""

    for(var i in this.normalised_tokens) {
      
      this.table_html += "<tr>";

      var columnOrders = [0, 6, 1, 2, 3, 4, 5];

      for(var j in columnOrders) {
        var jj = columnOrders[j];
        //if(j > 5) break; // Ignore columns after the 6th
        var td = "<td>"


        var val = this.normalised_tokens[i][jj];
        if(jj == T_CLASS || jj == T_CORRECT_CLASS) {
          val = "<span class='tag' style='background: " + class_colours[val] + "'>" + val + "</span>";
        }
        if(jj == T_SOURCE) {
          val = "<span class='tag' style='background: " + source_colours[val] + "'>" + val + "</span>";
        }
        if(jj == T_CORRECT) {
          val = val == true ? "<span class='yes'>Yes</span>" : "<span class='no'>No</span>";
        }

        td+=val;
        td += "</td>";
        this.table_html += td;
      }
      this.table_html+= "</tr>";

    }


    this.document_table_html = "";

    for(var i in normalised_documents) {
      var tr = "<tr><td>" + normalised_documents[i][0] + "</td><td>";

      var td = "";
      for(var j in normalised_documents[i][1]) {
        var obj = normalised_documents[i][1][j];
        if(Array.isArray(obj)) {
          td += "<span class=\"" + (obj[T_CORRECT] ? "correct" : "error") + 
                "\" data-original=\"" + obj[T_TOKEN] + 
                "\" title=\"" + "Original: " + obj[T_TOKEN] + "\nActual: " + obj[T_ACTUAL] + "\n----------------------------------\nClass: " + obj[T_CLASS] + "\nActual Class: " + obj[T_CORRECT_CLASS] +
                "\n----------------------------------\nSource: " + obj[T_SOURCE] + "\">" + obj[T_PREDICTION] + "</span> ";
        } else {
          td += obj + " ";
        }
      }



      
      tr += td + "</td></tr>";
      this.document_table_html += tr;
    }

  }



  // Show the button for this method.
  show() {
    this.button.removeClass("hidden");
  }

  // Hide the button for this method.
  hide() {
    this.button.addClass("hidden");
    this.button.removeClass("active");
  }

  getGraphData() {

    var totalTokens = this.normalised_tokens.length;
    var totals = {};
    var corrects = {};
    var source_corrects = {};
    var source_totals = {};
    var class_corrects = {};
    var class_totals   = {};

    for(var n of class_labels) {
      totals[n] = 0;
      corrects[n] = 0;
      class_totals[n] = 0;
      class_corrects[n] = 0;
    }
    for(var n of source_labels) {
      source_totals[n] = 0;
      source_corrects[n] = 0;
    }


    for(var n in this.normalised_tokens){ 
      totals[this.normalised_tokens[n][T_CORRECT_CLASS]] += 1   
      corrects[this.normalised_tokens[n][T_CORRECT_CLASS]] += (this.normalised_tokens[n][T_CORRECT] == true ? 1 : 0);   
      source_totals[this.normalised_tokens[n][T_SOURCE]] += 1   
      source_corrects[this.normalised_tokens[n][T_SOURCE]] += (this.normalised_tokens[n][T_CORRECT] == true ? 1 : 0);     
      class_totals[this.normalised_tokens[n][T_CORRECT_CLASS]] += 1   
      class_corrects[this.normalised_tokens[n][T_CORRECT_CLASS]] += (this.normalised_tokens[n][T_CLASS_IS_CORRECT] == true ? 1 : 0);    
    }

    var tokenClassGraphData = new Array(class_labels.size);
    var sourceGraphData = new Array(source_labels.size);
    var classifierPredictionsGraphData = new Array(class_labels.size);
    var i = 0;

    for(var c of class_labels) {
      tokenClassGraphData[i] = [c, totals[c], corrects[c]];
      classifierPredictionsGraphData[i] = [c, class_totals[c], class_corrects[c]]
      i++;
    }
    var i = 0;
    for(var s of source_labels) {
      sourceGraphData[i] = [s, source_totals[s], source_corrects[s]];
      i++;
    }

    return [tokenClassGraphData, sourceGraphData, classifierPredictionsGraphData];
  }

  // Display the results of this method.
  displayResults() {

    resultsTable.hideResultsTable();


    method_dropdown_button.blur();

    method_dropdown_items.children().removeClass("active");
    this.button.addClass("active");

    checkPrevNextButtons();

    method_dropdown_current.html(this.name)

    var sortingOrder  = predictions_datatable.order();
    var sortingOrder2 = documents_datatable.order();

    predictions_datatable.destroy();
    documents_datatable.destroy();
  
    si_experiment_name.html(this.name);
    var s = this.scores;
    si_fscore.html(s["F1"].toFixed(4));
    si_precision.html(s["Precision"].toFixed(4));
    si_recall.html(s["Recall"].toFixed(4));
    si_total_correct.html(s["Correct"]);
    si_total_incorrect.html(s["Incorrect"]);

    
    // for(var i in this.normalised_tokens) {
      
    //   var tr = document.createElement("TR");

    //   var columnOrders = [0, 6, 1, 2, 3, 4, 5];

    //   for(var j in columnOrders) {
    //     var jj = columnOrders[j];
    //     //if(j > 5) break; // Ignore columns after the 6th
    //     var td = document.createElement("TD");


    //     var val = this.normalised_tokens[i][jj];
    //     if(jj == T_CLASS || jj == T_CORRECT_CLASS) {
    //       val = "<span class='tag' style='background: " + class_colours[val] + "'>" + val + "</span>";
    //     }
    //     if(jj == T_SOURCE) {
    //       val = "<span class='tag' style='background: " + source_colours[val] + "'>" + val + "</span>";
    //     }
    //     if(jj == T_CORRECT) {
    //       val = val == true ? "<span class='yes'>Yes</span>" : "<span class='no'>No</span>";
    //     }

    //     td.innerHTML = val;
    //     tr.appendChild(td);
    //   }
    //   pt_tbody[0].appendChild(tr);

    // }


    pt_tbody.html(this.table_html);
    dt_tbody.html(this.document_table_html);
    
    predictions_datatable = makeDataTable();
    predictions_datatable.order(sortingOrder).columns.adjust().draw();

    documents_datatable = makeDocumentsDataTable();
    documents_datatable.order(sortingOrder2).columns.adjust().draw();



    classifierPredictionsGraph.update(this.classifierPredictionsGraphData, this.normalised_tokens.length);
    tokenClassGraph.update(this.tokenClassGraphData, this.normalised_tokens.length);
    sourceGraph.update(this.sourceGraphData, this.normalised_tokens.length);
  }
}