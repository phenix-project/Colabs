from __future__ import division, print_function
from libtbx import group_args

#  script to edit AlphaFold2_with_Manual_Template.ipynb and create
#  AlphaFold2.ipynb
#  Reason: no convenient way to make a link that has just these small
#  differences

input_file = 'AlphaFold2_with_Manual_Template.ipynb'
output_file = 'AlphaFold2.ipynb'
text = open(input_file).read()
print("Read text from %s" %(input_file))

# Here are the edits...

def get_replacement_group(lines, new_lines):

  return group_args(group_args_type = 'line group',
     lines_to_replace = lines,
     new_lines = new_lines,)

def apply_replacement_group(text, rg):

  target_strings = []
  for line in rg.lines_to_replace:
    target_strings.append(line.strip().replace(" ","").replace("\t",""))
  n = len(target_strings)
  if not n:
    return text

  new_lines = []
  lines = text.splitlines()
  str_lines = []
  for line in lines:
    str_lines.append(line.strip().replace(" ","").replace("\t",""))

  i = 0
  while i < len(lines):
    if str_lines[i : i + n] == target_strings:

      new_lines += rg.new_lines
      i += n
    else: # keep original
      new_lines.append(lines[i].rstrip())
      i += 1
  return "\n".join(new_lines)



replacement_groups = []

replacement_groups.append(get_replacement_group(
            ['''use_templates = False'''],
            ['''use_templates = True'''],))

replacement_groups.append(get_replacement_group(
            ['''"jobname = 'af_with_template' \\n",'''],
            ['''        "jobname = 'af_auto_templates' \\n",'''],))

replacement_groups.append(get_replacement_group(
            ['''"<a href=\\"https://colab.research.google.com/github/phenix-project/Colabs/blob/main/alphafold2/AlphaFold2_with_Manual_Template.ipynb\\" target=\\"_parent\\"><img src=\\"https://colab.research.google.com/assets/colab-badge.svg\\" alt=\\"Open In Colab\\"/></a>"'''],
            ['''         "<a href=\\"https://colab.research.google.com/github/phenix-project/Colabs/blob/main/alphafold2/AlphaFold2.ipynb\\" target=\\"_parent\\"><img src=\\"https://colab.research.google.com/assets/colab-badge.svg\\" alt=\\"Open In Colab\\"/></a>"''']))

replacement_groups.append(get_replacement_group(
             ['''"#AlphaFold2 with a Template (Phenix version)\\n",'''],
             ['''        "#AlphaFold2 (Phenix version)\\n",''']))

replacement_groups.append(get_replacement_group(
             ['''"name": "AlphaFold2_with_Manual_Template.ipynb",'''],
             ['''        "name": "AlphaFold2.ipynb",'''],))

replacement_groups.append(get_replacement_group(
            ['''"<a href=\"https://colab.research.google.com/github/phenix-project/Colabs/blob/main/alphafold2/AlphaFold2_with_Manual_Template.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"'''],
            ['''        "<a href=\"https://colab.research.google.com/github/phenix-project/Colabs/blob/main/alphafold2/AlphaFold2.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"''']))

#  replacement_groups.append(get_replacement_group(
#             [ '"use_templates = True #@param {type:\\"boolean\\"}\\n",'],
#             ['        "use_templates = False \\n",']))

replacement_groups.append(get_replacement_group(
             ['''"<b> HOW TO GET YOUR ALPHAFOLD MODEL USING A TEMPLATE</b>\\n",'''],
             ['''        "<b> HOW TO GET YOUR ALPHAFOLD MODEL</b>\\n",'''],))



replacement_groups.append(get_replacement_group(
             ['''        "- <font color='black'>\\n",''',
              '''"... Wait a minute or two for the \\"Choose files\\" button to appear below in a box labelled \\"Upload your .cif format template model here when button appears\\"\\n",''',
        '''"</font>\\n",''',
        '''"\\n",''',
        '''"- <b><font color='green'>\\n",''',
        '''"Upload your template (preliminary model). Must be an mmCIF file. To convert from pdb format use this converter: https://mmcif.pdbj.org/converter/\\n",''',
        '''"</font></b>\\n",''',
        '''"\\n",'''],
            []))

replacement_groups.append(get_replacement_group(
            ['''"... Wait a few more minutes for your result\\n",'''],
            ['''        "... Wait a few minutes for your result\\n",''',]))

replacement_groups.append(get_replacement_group(
            ['''        "#@title *Upload your .cif format template model here when button appears*\\n",''',
        '''"supply_manual_templates = True\\n",'''],
            ['''        "#@title Finding templates...\\n",''',
        '''"supply_manual_templates = False\\n",'''],))

replacement_groups.append(get_replacement_group(
        ['''"if not use_templates:\\n",'''],
        ['''        "if (not use_templates) or (not template_hits):\\n",''']))

replacement_groups.append(get_replacement_group(
 ['''        "  if template_paths is None:\\n",''',
 '''        "    template_features = mk_mock_template(query_sequence * homooligomer)\\n",'''],
 ['''        "  if template_paths is None:\\n",''',
 '''        "    template_features = mk_mock_template(query_sequence * homooligomer)\\n",''',
  '''        "    use_templates = False\\n",''']))

for rg in replacement_groups:
  print(rg)
  text = apply_replacement_group(text, rg)

# End of the edits...

f = open(output_file,'w')
print(text, file = f)
f.close()
print("Wrote to %s with changes" %(output_file))



