package sortAlpha;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

public class Main {
	static List<String> lines;

	public static void main(String[] args) throws IOException {
		//file input
		lines = (Files.readAllLines(Paths.get("C:/Users/Prog/Desktop/GroceryListPredictor/webapp/allproducts2.txt")));
		//sort and remove duplicates
		Set<String> singlelines = new TreeSet<String>(lines);
		//file output
		FileWriter writer = new FileWriter("C:/Users/Prog/Desktop/GroceryListPredictor/webapp/sortedProducts.txt");
		for (String str : singlelines) {
			writer.write(str + System.lineSeparator());
		}
		writer.close();
	}
}
