class Dataset {
  constructor(name, description) {
    this.name = name;
    this.description = description || "(random)";
    if(this.name == "Average") {
      this.description = "";
    }

    var ls = document.createElement("LI");
    var lsa = document.createElement("A");
    lsa.innerHTML = this.name + "<br/><span class='set-name'>" + this.description + "&nbsp;";
    ls.appendChild(lsa);
    sets_ul[0].appendChild(ls);

    this.button = $(ls);
    var t = this;
    this.button.on('click', function() {
      t.displayMethods();
    });
    this.methods = []; // An array of methods, such as "None_None_None".
  }

  // Display the buttons of the methods associated with this dataset.
  displayMethods() {
    var ii = getCurrentMethodIndex();
    for(var d in datasets) {
      datasets[d].hideMethods();
    }
    dataset_number.html(this.name);
    sets_ul.children().removeClass("active");
    this.button.addClass("active");

    for(var i in this.methods) {
      this.methods[i].show();
    }

  
    try {
      clickMethodWithIndex(ii);
    } catch(err) {
      clickMethodWithIndex(0);
    }   
      
    
  }

  hideMethods() {
    for(var i in this.methods) {
      this.methods[i].hide();
    }
  }

  // Add a method to this dataset's list of methods.
  addMethod(method) {
    this.methods.push(method);
  }
}